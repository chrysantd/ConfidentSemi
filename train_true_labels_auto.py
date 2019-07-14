import argparse
import torch
import numpy as np
import os
import os.path as osp
import pandas as pd
import time
from sklearn.metrics import accuracy_score,log_loss
from training.supervised_learning import SupervisedLearning
from dataset_given_label import SimpleDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import logging
import shutil
from sklearn.metrics import accuracy_score,log_loss

import traceback
import torch.multiprocessing as multiprocessing

from torch.utils.data.sampler import SubsetRandomSampler
from utils import *
from torch.utils.data import ConcatDataset
import bisect

import loader_svhn as svhn
import loader_cifar_zca as cifar_zca
import loader_cifar as cifar

import math

from autoaugment import CIFAR10Policy, SVHNPolicy, Cutout
from adamw import AdamW

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Semi Supervised")

    
    
    parser.add_argument("--snapshot-dir", type=str, default='snapshots',
                        help="snapshot directory")
    parser.add_argument("--use-gpu",default=False,action='store_true', 
                        help="whether to use gpu")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which gps to use")
    parser.add_argument("--dataset", type=str, default='svhn',choices=['svhn','cifar10'],
                        help="dataset type")
    parser.add_argument("--model", type=str, default='wide28',choices=['wide22','wide28','wide28_2'],
                        help="dataset type")     
    parser.add_argument('--boundary',default=0, type=int, help='different label/unlabel division [0,9]')

    #model and loader settings
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Number of images sent to the network in one epoch.")
    parser.add_argument("--eval-batch-size", type=int, default=500,
                        help="Number of eval data sent to the network in one epoch.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="number of workers for multithread dataloading.")        
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, required=False,
                        help="pretrained model for initialization")
    parser.add_argument("--std", type=float, default=0.05,
                        help="std of CNN")
    parser.add_argument("--label-num",type=int,default=1000,
                        help="labeled data size")  
    parser.add_argument("--val-num",type=int,default=7320,
                        help="valid data size")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument('--alpha', default=0.75, type=float)

    #optimizer settings
    
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--min-epoch", type=int, default=5,
                        help="Max training epoch.")
    parser.add_argument("--max-epoch", type=int, default=12000,
                        help="Max training epoch.")
    parser.add_argument("--save-epoch", type=int, default=0,
                        help="epoch inteval for checkpoint, set 0 to forbid")
    parser.add_argument("--evaluate-epoch", type=int, default=10,
                        help="epoch inteval for evaluation, set 0 to forbid")
    parser.add_argument("--eval-method", type=str, default='acc',choices=['loss','acc'],
                        help="epoch inteval for evaluation, set 0 to forbid")  



    return parser.parse_args()


args = get_arguments()

def adjust_learning_rate_adam(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""
    
    boundary = [args.max_epoch//5 * 4]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    #print('Learning rate: %f'%lr)
    #print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=20.0):
    """Sets the weights for the consistency loss"""
    if epoch > end_ep:
        weight_cl = end_w
    elif epoch < init_ep:
        weight_cl = init_w
    else:
        T = float(epoch - init_ep)/float(end_ep - init_ep)
        #weight_mse = T * (end_w - init_w) + init_w #linear
        weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
    #print('Consistency weight: %f'%weight_cl)
    return weight_cl


def train(trainloader,model,optimizer,loss_fn,epoch,logger=None):
    model.train()    

    for x , y, _ in trainloader:
        optimizer.zero_grad()
        y = y.long()
        if args.use_gpu:                
            x = x.cuda()
            y = y.cuda()
        y_pred = model(x)

        train_loss = loss_fn(y_pred,y)
        train_loss.backward()
        
        optimizer.step()
        if logger is not None:
            logger.info('epoch{} : train loss = {:.4f} '.format(epoch, train_loss.item()))


def train_with_consistence_loss(trainloader,model,optimizer,loss_fns,epoch,logger=None):
    model.train()
    len_iter = len(iter(trainloader))
    #global_step = epoch * len_iter

    ce_fn, mse_fn = loss_fns
    
    for x , y, x1 in trainloader:
        optimizer.zero_grad()
        y = y.long()
        if args.use_gpu:                
            x = x.cuda()
            y = y.cuda()
            x1 = x1.cuda()
        y_pred = model(x)
        with torch.no_grad():
            y_pred1 = model(x1)

        loss_cl = mse_fn(y_pred,y_pred1) / float(args.num_classes)
        loss_ce = ce_fn(y_pred,y)

    
        weight_cl = cal_consistency_weight(epoch, end_ep=(args.max_epoch//2.5), end_w=1.0)
        #global_step = global_step + 1

        train_loss = loss_ce + weight_cl * loss_cl

        train_loss.backward()
        
        optimizer.step()
        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item()))


def train_with_unlabel(confident_loader,unconfident_loader,model,optimizer,loss_fns,epoch,logger=None):
    model.train()

    conf_iter = iter(confident_loader)
    unconf_iter = iter(unconfident_loader)

    len_iter = max(len(conf_iter),len(unconf_iter))
    #global_step = epoch * len_iter

    ce_fn, mse_fn = loss_fns

    #sample_ce_fn = torch.nn.CrossEntropyLoss(reduction='none').cuda()

    for i in range(len_iter):
        
        try:
            input_conf, target_conf, input1_conf = next(conf_iter)
        except StopIteration:
            conf_iter = iter(confident_loader)
            input_conf, target_conf, input1_conf = next(conf_iter)
        
        try:
            input_unconf, _, input1_unconf = next(unconf_iter)
        except StopIteration:
            unconf_iter = iter(unconfident_loader)
            input_unconf, _, input1_unconf = next(unconf_iter)
        
        if args.use_gpu:
            input_conf = input_conf.cuda()
            target_conf = target_conf.long().cuda()
            input1_conf = input1_conf.cuda()
            input_unconf = input_unconf.cuda()
            input1_unconf = input1_unconf.cuda()
                
        sc = input_conf.shape[0]
        su = input_unconf.shape[0]

        input_concat_var = torch.cat([input_conf,input_unconf])
        input1_concat_var = torch.cat([input1_conf,input1_unconf])

        output = model(input_concat_var)
        with torch.no_grad():
            output1 = model(input1_concat_var)

        loss_cl = mse_fn(output,output1) / float(args.num_classes)
        loss_ce = ce_fn(output[:sc], target_conf)
        
    
        weight_cl = cal_consistency_weight(epoch, end_ep=(args.max_epoch//2.5), end_w=1.0)
        #weight_cl = 1.0
        #global_step = global_step + 1

        train_loss = loss_ce + weight_cl * loss_cl

        optimizer.zero_grad()
        train_loss.backward()        
        optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item()))


def train_with_mix(trainloader,model,optimizer,loss_fns,epoch,logger=None):
    model.train()
    #global_step = epoch * len_iter
    ce_fn, mse_fn = loss_fns

    iter_loader = iter(trainloader)
    len_iter = len(iter_loader)
    
    for i in range(len_iter):

        try:
            x , y, x1 = next(iter_loader)
        except StopIteration:
            iter_loader = iter(trainloader)
            x , y, x1 = next(iter_loader) 

        size = x.size(0)
        y = y.long()

        if args.use_gpu:                
            x = x.cuda()
            y = y.cuda()
            x1 = x1.cuda()    
    
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        idx = torch.randperm(size * 2)

        x_concat = torch.cat([x,x1])
        y_concat = torch.cat([y,y])

        input_a, input_b = x_concat, x_concat[idx]
        target_a, target_b = y_concat, y_concat[idx]

        mixed_input = l * input_a + (1 - l) * input_b

        mixed_pred = model(mixed_input)

        #loss_ce = ce_fn(y_pred,y)

        loss_ce = l * ce_fn(mixed_pred,target_a) + (1 - l) *ce_fn(mixed_pred,target_b)
        
        train_loss = loss_ce

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f}'.format(epoch, loss_ce.item()))




def evaluate(testloader,model,logger=None,prefix=''):
    model.eval()

    y_true_all = np.zeros(len(testloader.sampler))
    y_pred_all = np.zeros(len(testloader.sampler))
    y_pred_prob_all = np.zeros((len(testloader.sampler) , args.num_classes))
    idx = 0

    for index, batch in enumerate(testloader):
        x, y, _ = batch
        if args.use_gpu:
            x = x.cuda()

        y_true = y.numpy().reshape(-1)
        y_pred_prob = torch.softmax(model(x),1).cpu().data.numpy()
        n_pixel = y_true.shape[0]     

        y_true_all[idx:idx + n_pixel] = y_true            
        y_pred_prob_all[idx:idx + n_pixel,:] =  y_pred_prob     
        y_pred_all[idx:idx + n_pixel] = y_pred_prob.argmax(axis=1)

        idx +=  n_pixel

    logloss = log_loss(y_true_all,y_pred_prob_all,labels=np.arange(args.num_classes))
    acc = accuracy_score(y_true_all,y_pred_all)
    if logger is not None:
        logger.info("{}: logloss={:.4f},acc={:.4f}".format(prefix,logloss,acc))
    return logloss,acc


def main():
    print(args)

    if not osp.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if args.use_gpu:
        cudnn.enabled = True 
        cudnn.benchmark = True
        torch.cuda.set_device(args.gpu)

    save_path = osp.join(args.snapshot_dir,'sup_model.pth')
  
    labeled_size =  args.label_num + args.val_num
    
    if args.dataset == 'cifar10':
        dataloader = cifar.CIFAR10
        num_classes = 10
        data_dir = '../cifar10_data/'

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470,  0.2435,  0.2616])

        # transform is implemented inside zca dataloader 
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
            transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
            normalize
        ])
        

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])


     
    elif args.dataset == 'svhn':
        dataloader = svhn.SVHN
        num_classes = 10
        data_dir = '../svhn_data/'
 
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        transform_train = transforms.Compose([
            SVHNPolicy(), 
			transforms.ToTensor(), 
            Cutout(n_holes=1, length=20), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
            normalize
        ])
 
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]) 

    
    labelset = dataloader(root=data_dir, split='label', download=True, 
        transform=transform_train, boundary=args.boundary, label_data_number = args.label_num, val_data_number = args.val_num)    

    unlabelset = dataloader(root=data_dir, split='unlabel', download=True, 
        transform=transform_train, boundary=args.boundary, label_data_number = args.label_num, val_data_number = args.val_num)

    validset = dataloader(root=data_dir, split='valid', download=True, 
        transform=transform_test, boundary=args.boundary, label_data_number = args.label_num, val_data_number = args.val_num)

    testset = dataloader(root=data_dir, split='test', download=True, 
        transform=transform_test, label_data_number = args.label_num, val_data_number = args.val_num)

    
    logger = logging.getLogger("main")
    file_handler = logging.FileHandler(osp.join(args.snapshot_dir,'sup_logs.txt'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    trainloader = torch.utils.data.DataLoader(labelset, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    unlabelloader = torch.utils.data.DataLoader(unlabelset, batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(validset, batch_size=args.eval_batch_size,shuffle=False,num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset,batch_size=args.eval_batch_size,num_workers=args.num_workers )

    model=create_model(args.num_classes,args.model)

    if args.use_gpu:
        model = model.cuda()

    best_val_loss = float('inf')
    best_val_acc = .0
    best_val_epoch = 0 

    best_val_loss,best_val_acc = evaluate(valloader,model,logger=logger,prefix='val')

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_fns = (torch.nn.CrossEntropyLoss(),torch.nn.MSELoss())

    for epoch in range(args.max_epoch):
        adjust_learning_rate_adam(optimizer,epoch)
        train_with_mix(trainloader,model,optimizer,loss_fns,epoch,logger)
        #train_with_unlabel(trainloader,unlabelloader,model,optimizer,loss_fns,epoch,logger)
        
        if args.evaluate_epoch> 0 and epoch % args.evaluate_epoch == 0:
            val_loss,val_acc = evaluate(valloader,model,logger=logger,prefix='val')
            
            if args.eval_method == 'acc' and (val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss)):
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_epoch = epoch
                torch.save(model.state_dict(), save_path)
                    
            elif args.eval_method == 'loss' and val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_val_epoch = epoch                    
                torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    evaluate(valloader,model,logger=logger,prefix='val')
    evaluate(testloader,model,logger=logger,prefix='test')

            

if __name__ == '__main__':
    main()