import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path as osp
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import logging
import shutil
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error

import traceback
import torch.multiprocessing as multiprocessing

from utils import *
from torch.utils.data import ConcatDataset
import random

from torch.utils.data.sampler import Sampler
import itertools
import math

import bisect
import loader_cifar as cifar

from autoaugment import CIFAR10Policy, SVHNPolicy, Cutout
from adamw import AdamW
from loader_cifar2 import CIFAR10_labeled,train_val_split

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Semi Supervised")
    

    parser.add_argument("--round", type=int, default=100,
                        help="max number of iterations")              
    parser.add_argument("--label-num",type=int,default=4000,
                        help="labeled data size") 
    parser.add_argument("--val-num",type=int,default=5000,
                        help="valid data size")                      
    parser.add_argument("--dir", type=str, default='snapshots',
                        help="snapshot directory")
    parser.add_argument("--dataset", type=str, default='cifar10',choices=['cifar10'],
                        help="dataset type")
    parser.add_argument("--model", type=str, default='wide28_2',choices=['wide28_2'],
                        help="dataset type")     
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    parser.add_argument("--use-gpu",default=False,action='store_true', 
                        help="whether to use gpu")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which gps to use")    
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--K', default=10, type=int)
    parser.add_argument("--auto",default=False,action='store_true', 
                        help="whether to use soft labels or pseudo labels")
    parser.add_argument("--mix",default=False,action='store_true', 
                        help="whether to use soft labels or pseudo labels")
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)
 

    #model and loader settings
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Number of images sent to the network in one epoch.")
    parser.add_argument("--batch-size2", type=int, default=192,
                        help="Number of images sent to the network in one epoch.")
    parser.add_argument("--eval-batch-size", type=int, default=1000,
                        help="Number of eval data sent to the network in one epoch.")    
    parser.add_argument("--num-classes", type=int, default=10,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers.")
    parser.add_argument("--restore-from", type=str, required=False,
                        help="pretrained model for child models")
    parser.add_argument("--init", type=str, required=False,
                        help="pretrained model for initialization") 
    parser.add_argument("--init1", type=str, required=False,
                        help="pretrained model for initialization") 
    parser.add_argument("--init2", type=str, required=False,
                        help="pretrained model for initialization") 
    parser.add_argument("--sample-method", type=str, default='max',choices=['max','sample'],
                        help="sample method")
    parser.add_argument("--drop-rate",type=float, default=0.1,
                        help="drop rate of training data")   
    parser.add_argument("--schedule", type=str, default='exp',choices=['exp','linear'],
                        help="schedule type")     
    parser.add_argument("--scale",type=float, default=0.15,
                        help="round ratio of confident data")
    #optimizer settings
    
    parser.add_argument("--lr", type=float, default=0.002,
                        help="Base learning rate for training with polynomial decay.")    
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--min-epoch", type=int, default=0,
                        help="Max training epoch.")
    parser.add_argument("--steps", type=int, default=200000,
                        help="Max training epoch.")
    parser.add_argument("--steps1", type=int, default=50000,
                        help="Max training epoch.")
    parser.add_argument("--steps2", type=int, default=50000,
                        help="Max training epoch.")
    parser.add_argument("--stop-steps1", type=int, default=10000,
                        help="early stop steps.")
    parser.add_argument("--stop-steps2", type=int, default=5000,
                        help="early stop steps.")                        
    parser.add_argument("--evaluate-epoch", type=int, default=1,
                        help="epoch inteval for evaluation, set 0 to forbid")
    parser.add_argument("--eval-method", type=str, default='acc',choices=['loss','acc'],
                        help="epoch inteval for evaluation, set 0 to forbid")
    
    return parser.parse_args()

args = get_arguments()


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu

ce_fn = torch.nn.CrossEntropyLoss()
ce_fn2 = torch.nn.CrossEntropyLoss(reduction='none')
mse_fn = torch.nn.MSELoss()
kl_fn = torch.nn.KLDivLoss(reduction='batchmean')
l1_fn = torch.nn.L1Loss()

semi_loss = SemiLoss()


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.tmp_model = create_model(args.num_classes,args.model)
        if args.use_gpu:
            self.tmp_model = self.tmp_model.cuda()
        self.wd = 0.02 * args.lr

        self.ema_model.load_state_dict(self.model.state_dict())

    def step(self, bn=False):
        if bn:
            # copy batchnorm stats to ema model
            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                tmp_param.data.copy_(ema_param.data.detach())

            self.ema_model.load_state_dict(self.model.state_dict())

            for ema_param, tmp_param in zip(self.ema_model.parameters(), self.tmp_model.parameters()):
                ema_param.data.copy_(tmp_param.data.detach())
        else:
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.alpha)
                ema_param.data.add_(param.data.detach() * one_minus_alpha)
                # customized weight decay
                param.data.mul_(1 - self.wd)


def train_model1_mix(label_loader,unlabel_loader,model,optimizer,ema_model,ema_optimizer,steps_per_epoch,epoch,logger=None):
    model.train()

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    
    for i in range(steps_per_epoch):

        try:
            xl, yl, _ = next(label_iter)
        except:
            label_iter = iter(label_loader)
            xl, yl, _ = next(label_iter)
        
        try:
            xu1, yu, xu2 = next(unlabel_iter)
        except:
            unlabel_iter = iter(unlabel_loader)
            xu1, yu, xu2 = next(unlabel_iter)

        yl = yl.long()
        yu = yu.long()

        if args.use_gpu:                
            xl = xl.cuda()            
            xu1 = xu1.cuda()
            xu2 = xu2.cuda()
            yl = yl.cuda()
            yu = yu.cuda()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        x_concat = torch.cat([xl,xu1,xu2])
        y_concat = torch.cat([yl,yu,yu])

        '''if args.use_gpu:
            y_concat = torch.zeros(y_concat.size(0), args.num_classes).cuda().scatter_(1, y_concat.view(-1,1), 1) 
        else:
            y_concat = torch.zeros(y_concat.size(0), args.num_classes).scatter_(1, y_concat.view(-1,1), 1)'''

        size = x_concat.size(0)
        idx = torch.randperm(size)
        input_a, input_b = x_concat, x_concat[idx]
        target_a, target_b = y_concat, y_concat[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        #mixed_target = l * target_a + (1 - l) * target_b
        mixed_output = model(mixed_input)

        #loss = mse_fn(torch.softmax(mixed_output,1),mixed_target)

        loss = l * ce_fn(mixed_output,target_a)  + (1-l) * ce_fn(mixed_output, target_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss = {:.4f}'.format(epoch, loss.item()))
    
    ema_optimizer.step(bn=True)




def train_model1(label_loader,unlabel_loader,model,optimizer,ema_model,ema_optimizer,steps_per_epoch,epoch,logger=None):
    model.train()

    label_iter = iter(label_loader)
    unlabel_iter = iter(unlabel_loader)
    
    for i in range(steps_per_epoch):

        try:
            xl, yl, _ = next(label_iter)
        except:
            label_iter = iter(label_loader)
            xl, yl, _ = next(label_iter)
        
        try:
            xu1, yu, xu2 = next(unlabel_iter)
        except:
            unlabel_iter = iter(unlabel_loader)
            xu1, yu, xu2 = next(unlabel_iter)

        yl = yl.long()
        yu = yu.long()

        if args.use_gpu:                
            xl = xl.cuda()            
            xu1 = xu1.cuda()
            xu2 = xu2.cuda()
            yl = yl.cuda()
            yu = yu.cuda()

        x_concat = torch.cat([xl,xu1,xu2])
        y_concat = torch.cat([yl,yu,yu])

        output = model(x_concat)

        loss = ce_fn(output,y_concat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss = {:.4f}'.format(epoch, loss.item()))
    
    ema_optimizer.step(bn=True)




def train_init_mix(label_loader,model,optimizer,ema_optimizer,steps_per_epoch,epoch,logger=None):
    model.train()

    label_iter = iter(label_loader)
    
    for i in range(steps_per_epoch):

        try:
            xl, yl, _ = next(label_iter)
        except:
            label_iter = iter(label_loader)
            xl, yl, _ = next(label_iter)
     

        yl = yl.long()

        if args.use_gpu:                
            xl = xl.cuda()
            yl = yl.cuda()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
                
        '''if args.use_gpu:
            yl = torch.zeros(yl.size(0), args.num_classes).cuda().scatter_(1, yl.view(-1,1), 1) 
        else:
            yl = torch.zeros(yl.size(0), args.num_classes).scatter_(1, yl.view(-1,1), 1)'''

        size = xl.size(0)
        idx = torch.randperm(size)
        
        input_a, input_b = xl, xl[idx]
        target_a, target_b = yl, yl[idx]

        mixed_input = l * input_a + (1 - l) * input_b

        '''mixed_target = l * target_a + (1 - l) * target_b
        mixed_output = model(mixed_input)
        loss = mse_fn(torch.softmax(mixed_output,1),mixed_target)'''

        mixed_output = model(mixed_input)
        loss = l * ce_fn(mixed_output,target_a) + (1-l) * ce_fn(mixed_output,target_b)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss = {:.4f}'.format(epoch, loss.item()))    
    
    ema_optimizer.step(bn=True)



def train_init(label_loader,model,optimizer,ema_optimizer,steps_per_epoch,epoch,logger=None):
    model.train()
    label_iter = iter(label_loader)    
    
    for i in range(steps_per_epoch):
        try:
            xl, yl, _ = next(label_iter)
        except:
            label_iter = iter(label_loader)
            xl, yl, _ = next(label_iter)

        yl = yl.long()        

        if args.use_gpu:                
            xl = xl.cuda()            
            yl = yl.cuda()
            
        output = model(xl)

        loss = ce_fn(output,yl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss = {:.4f}'.format(epoch, loss.item()))
    
    ema_optimizer.step(bn=True)



def train_model2(trainloader,model,optimizer,epoch,logger=None):
    model.train()
    
    for x , y, _ in trainloader:

        size = x.size(0)
        y = y.long()

        if args.use_gpu:                
            x = x.cuda()
            y = y.cuda()

        output = model(x)

        loss_ce = ce_fn(output,y)

        optimizer.zero_grad()
        loss_ce.backward()
        optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f}'.format(epoch, loss_ce.item()))


def evaluate(testloader,model,logger=None,prefix=''):
    model.eval()
    with torch.no_grad(): 
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


def cal_consistency_weight(epoch, init_ep=0, end_ep=150, init_w=0.0, end_w=1.0):
    if epoch > end_ep:
        weight = end_w
    elif epoch < init_ep:
        weight = init_w
    else:
        T = float(epoch - init_ep)/float(end_ep - init_ep)
        weight = T * (end_w - init_w) + init_w #linear
        #weight = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp    
    return weight


def predict_probs(model, loader):
    probs = np.zeros((len(loader.sampler),args.num_classes))
    model.eval()

    with torch.no_grad():

        idx = 0              

        for x, _, _ in loader:
            if args.use_gpu:
                x = x.cuda()            

            preds = torch.softmax(model(x),1)
            
            size = preds.size()[0]      

            probs[idx:idx + size] = preds.cpu().data.numpy().reshape(-1,args.num_classes)
            idx +=  size
        
        
        probs = probs / probs.sum(axis=1)[:,np.newaxis]

        return probs


def sample_by_probs(probs, round):
    size = probs.shape[0]
    classes = probs.shape[1]
    pseudo_labels = np.zeros((round,size))

    for i in range(size):
        pseudo_labels[:,i] = np.random.choice(range(classes),round,p=probs[i])

    pseudo_labels = pseudo_labels.astype(int)

    return pseudo_labels


def create_basic_stats_dataframe():
    df = pd.DataFrame([[]])
    df['iter'] = 0
    df['train_acc'] = np.nan
    df['valid_acc'] = np.nan
    df['valid_epoch'] = 0
    return df


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def adjust_learning_rate_adam(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    print(args)

    if not osp.exists(args.dir):
        os.makedirs(args.dir)

    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
        cudnn.enabled = True 
        cudnn.benchmark = True

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    np.random.seed(args.manualSeed)


    labeled_size =  args.label_num + args.val_num
    
    
    num_classes = 10
    data_dir = '../cifar10_data/'

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2470,  0.2435,  0.2616])

    # transform is implemented inside zca dataloader 
    dataloader = cifar.CIFAR10
    if args.auto:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
            transforms.RandomHorizontalFlip(), CIFAR10Policy(), 
            transforms.ToTensor(), 
            Cutout(n_holes=1, length=16), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128), # fill parameter needs torchvision installed from source
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),                 
            normalize
        ])
    

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    base_dataset = datasets.CIFAR10(data_dir, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(args.label_num/10))

    labelset = CIFAR10_labeled(data_dir, train_labeled_idxs, train=True, transform=transform_train)
    labelset2 = CIFAR10_labeled(data_dir, train_labeled_idxs, train=True, transform=transform_test)
    unlabelset = CIFAR10_labeled(data_dir, train_unlabeled_idxs, train=True, transform=transform_train)
    unlabelset2 = CIFAR10_labeled(data_dir, train_unlabeled_idxs, train=True, transform=transform_test)
    validset = CIFAR10_labeled(data_dir, val_idxs, train=True, transform=transform_test)
    testset = CIFAR10_labeled(data_dir, train=False, transform=transform_test)

    label_y = np.array(labelset.targets).astype(np.int32)
    unlabel_y =  np.array(unlabelset.targets).astype(np.int32)
    unlabel_num = unlabel_y.shape[0]

    label_loader = torch.utils.data.DataLoader(
        labelset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True)

    label_loader2 = torch.utils.data.DataLoader(
        labelset2,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True )

    unlabel_loader = torch.utils.data.DataLoader(
        unlabelset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True)
    
    unlabel_loader2 = torch.utils.data.DataLoader(
        unlabelset2,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True)

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True 
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True 
    )

    #initialize models
    model1 = create_model(args.num_classes,args.model)
    model2 = create_model(args.num_classes,args.model)
    ema_model = create_model(args.num_classes,args.model)    
    
    if args.use_gpu:
        model1 = model1.cuda()
        model2 = model2.cuda()
        ema_model = ema_model.cuda()     
    
    for param in ema_model.parameters():
        param.detach_()

    df = pd.DataFrame()
    stats_path = osp.join(args.dir,'stats.txt')    
    
    '''if prop > args.scale:
        prop = args.scale'''    
    
    optimizer1 = AdamW(model1.parameters(), lr=args.lr)

    if args.init1 and osp.exists(args.init1):        
        model1.load_state_dict(torch.load(args.init1, map_location='cuda:{}'.format(args.gpu))) 

    ema_optimizer = WeightEMA(model1,ema_model,alpha=args.ema_decay)

    if args.init and osp.exists(args.init):
        model1.load_state_dict(torch.load(args.init, map_location='cuda:{}'.format(args.gpu))) 

    _ , best_acc = evaluate(validloader,ema_model,prefix='val')
    
    best_ema_path = osp.join(args.dir,'best_ema.pth')
    best_model1_path = osp.join(args.dir,'best_model1.pth')
    best_model2_path = osp.join(args.dir,'best_model2.pth')
    init_path = osp.join(args.dir,'init_ema.pth')    
    init_path1 = osp.join(args.dir,'init1.pth')
    init_path2 = osp.join(args.dir,'init2.pth')
    torch.save(ema_model.state_dict(), init_path)
    torch.save(model1.state_dict(), init_path1)
    torch.save(model2.state_dict(), init_path2)
    torch.save(ema_model.state_dict(), best_ema_path)
    torch.save(model1.state_dict(), best_model1_path)         
    skip_model2 = False
    end_iter = False

    confident_indices = np.array([],dtype=np.int64)
    all_indices = np.arange(unlabel_num).astype(np.int64)
    #no_help_indices = np.array([]).astype(np.int64)
    pseudo_labels = np.zeros(all_indices.shape,dtype=np.int32)       

    steps_per_epoch = len(iter(label_loader))
    max_epoch = args.steps // steps_per_epoch

    logger = logging.getLogger('init')
    file_handler = logging.FileHandler(osp.join(args.dir,'init.txt'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    for epoch in range(max_epoch * 4 // 5):
        if args.mix:
            train_init_mix(label_loader,model1,optimizer1,ema_optimizer,steps_per_epoch,epoch,logger=logger)
        else:
            train_init(label_loader,model1,optimizer1,ema_optimizer,steps_per_epoch,epoch,logger=logger)

        if epoch % 10 == 0:
            val_loss,val_acc = evaluate(validloader,ema_model,logger,'valid')
            if val_acc >= best_acc:
                best_acc = val_acc            
                evaluate(testloader,ema_model,logger,'test')                                  
                torch.save(ema_model.state_dict(), init_path)
                torch.save(model1.state_dict(), init_path1)

    adjust_learning_rate_adam(optimizer1, args.lr * 0.2)

    for epoch in range(max_epoch // 5):
        if args.mix:
            train_init_mix(label_loader,model1,optimizer1,ema_optimizer,steps_per_epoch,epoch,logger=logger)
        else:
            train_init(label_loader,model1,optimizer1,ema_optimizer,steps_per_epoch,epoch,logger=logger)
        
        if epoch % 10 == 0:
            val_loss,val_acc = evaluate(validloader,ema_model,logger,'valid')
            if val_acc >= best_acc:
                best_acc = val_acc            
                evaluate(testloader,ema_model,logger,'test')                                  
                torch.save(ema_model.state_dict(), init_path)
                torch.save(model1.state_dict(), init_path1)

    ema_model.load_state_dict(torch.load(init_path))
    model1.load_state_dict(torch.load(init_path1))
    

    logger.info('init train finished')
    evaluate(validloader,ema_model,logger,'valid')  
    evaluate(testloader,ema_model,logger,'test')
    

    for i_round in range(args.round):
        mask = np.zeros(all_indices.shape,dtype=bool)
        mask[confident_indices] = True
        other_indices = all_indices[~mask]
        
        optimizer2 = AdamW(model2.parameters(), lr=args.lr)
        
        logger = logging.getLogger('model2_round_{}'.format(i_round))
        file_handler = logging.FileHandler(osp.join(args.dir,'model2_round_{}.txt'.format(i_round)))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
                         

        if args.auto:
            probs  = predict_probs(ema_model,unlabel_loader2)
        else:
            probs = np.zeros((unlabel_num,args.num_classes))
            for i in range(args.K):
                probs += predict_probs(ema_model, unlabel_loader)
            probs /= args.K
        
        pseudo_labels[other_indices] = probs.argmax(axis=1).astype(np.int32)[other_indices]
        #pseudo_labels = probs.argmax(axis=1).astype(np.int32)
        
        df2 = create_basic_stats_dataframe()
        df2['iter'] = i_round
        df2['train_acc'] = accuracy_score(unlabel_y,pseudo_labels)   
        df = df.append(df2, ignore_index=True)
        df.to_csv(stats_path,index=False)

        #phase2: train model2
        unlabelset.targets = pseudo_labels.copy()            
        trainset = ConcatDataset([labelset,unlabelset])        
        
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size2,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)
                    
        model2.load_state_dict(torch.load(init_path2))
        best_val_epoch = 0 
        best_model2_acc = 0

        steps_per_epoch = len(iter(trainloader))
        max_epoch2 = args.steps2 // steps_per_epoch
        
        for epoch in range(max_epoch2):              
            train_model2(trainloader,model2,optimizer2,epoch,logger=logger)

            val_loss,val_acc = evaluate(validloader,model2,logger,'val')

            if val_acc >= best_model2_acc:
                best_model2_acc = val_acc
                best_val_epoch = epoch
                torch.save(model2.state_dict(), best_model2_path)
                evaluate(testloader,model2,logger,'test')
            
            if (epoch - best_val_epoch) * steps_per_epoch > args.stop_steps2:
                break

        df.loc[df['iter'] == i_round,'valid_acc'] = best_model2_acc
        df.loc[df['iter'] == i_round,'valid_epoch'] = best_val_epoch
        df.to_csv(stats_path,index=False)

        model2.load_state_dict(torch.load(best_model2_path))
        logger.info('model2 train finished') 

        evaluate(trainloader, model2, logger, 'train')

        evaluate(validloader,model2,logger,'val')
        evaluate(label_loader2,model2,logger,'reward')        
        evaluate(testloader,model2,logger,'test')
        #phase3: get confidence of unlabeled data by labeled data, split confident and unconfident data
        '''if args.auto:
            probs  = predict_probs(model2,unlabel_loader2)
        else:
            probs = np.zeros((unlabel_num,args.num_classes))
            for i in range(args.K):
                probs += predict_probs(model2, unlabel_loader)
            probs /= args.K'''
        
        probs  = predict_probs(model2,unlabel_loader2)
        new_pseudo_labels = probs.argmax(axis=1)
        
        confidences =  probs[all_indices,pseudo_labels]

        if args.schedule == 'exp':
            confident_num = int((len(confident_indices) + args.label_num) * (1 +args.scale)) - args.label_num
        elif args.schedule == 'linear':
            confident_num = len(confident_indices) + int(unlabel_num * args.scale)

        old_confident_indices = confident_indices.copy()
        confident_indices = np.array([],dtype=np.int64)

        for j in range(args.num_classes):
            j_cands = (pseudo_labels==j)
            k_size = int(min(confident_num // args.num_classes, j_cands.sum()))
            logger.info('class: {}, confident size: {}'.format(j,k_size))
            if k_size > 0:
                j_idx_top = all_indices[j_cands][confidences[j_cands].argsort()[-k_size:]]
                confident_indices = np.concatenate((confident_indices, all_indices[j_idx_top]))

        '''new_confident_indices = np.intersect1d(new_confident_indices, np.setdiff1d(new_confident_indices, no_help_indices))
        new_confident_indices = new_confident_indices[(-confidences[new_confident_indices]).argsort()]
        confident_indices = np.concatenate((old_confident_indices, new_confident_indices))'''

        acc = accuracy_score(unlabel_y[confident_indices],pseudo_labels[confident_indices])        
        logger.info('confident data num:{}, prop: {:4f}, acc: {:4f}'.format(len(confident_indices), len(confident_indices)/len(unlabel_y), acc))

        '''if len(old_confident_indices) > 0:
            acc = accuracy_score(unlabel_y[old_confident_indices],pseudo_labels[old_confident_indices])        
            logger.info('old confident data prop: {:4f}, acc: {:4f}'.format(len(old_confident_indices)/len(unlabel_y), acc))

        acc = accuracy_score(unlabel_y[new_confident_indices],pseudo_labels[new_confident_indices])
        logger.info('new confident data prop: {:4f}, acc: {:4f}'.format(len(new_confident_indices)/len(unlabel_y), acc))'''

        #unlabelset.train_labels_ul = pseudo_labels.copy()
        confident_dataset = torch.utils.data.Subset(unlabelset,confident_indices)        

        #phase4: refine model1 by confident data and reward data
        #train_dataset = torch.utils.data.ConcatDataset([confident_dataset,labelset])

        logger = logging.getLogger('model1_round_{}'.format(i_round))
        file_handler = logging.FileHandler(osp.join(args.dir,'model1_round_{}.txt'.format(i_round)))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)        
        
        best_val_epoch = 0
        evaluate(validloader,ema_model,logger,'valid')
        evaluate(testloader,ema_model,logger,'test')

        optimizer1 = AdamW(model1.parameters(), lr=args.lr)

        confident_dataset = torch.utils.data.Subset(unlabelset,confident_indices)
        trainloader = torch.utils.data.DataLoader(
            confident_dataset,
            batch_size= args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True 
        )        

        #steps_per_epoch = len(iter(trainloader))
        steps_per_epoch = 200
        max_epoch1 = args.steps1 // steps_per_epoch
        

        for epoch in range(max_epoch1):            
            '''current_num = int(cal_consistency_weight( (epoch + 1) * steps_per_epoch, init_ep=0, end_ep=args.stop_steps1//2, init_w=start_num, end_w=end_num))            
            current_confident_indices = confident_indices[:current_num]
            logger.info('current num: {}'.format(current_num))'''
            if args.mix:
                train_model1_mix(label_loader,trainloader,model1,optimizer1,ema_model,ema_optimizer,steps_per_epoch,epoch,logger=logger)
            else:
                train_model1(label_loader,trainloader,model1,optimizer1,ema_model,ema_optimizer,steps_per_epoch,epoch,logger=logger)

            val_loss,val_acc = evaluate(validloader,ema_model,logger,'valid')
            if val_acc >= best_acc:
                best_acc = val_acc
                best_val_epoch = epoch
                evaluate(testloader,ema_model,logger,'test')                                 
                torch.save(model1.state_dict(), best_model1_path)     
                torch.save(ema_model.state_dict(), best_ema_path)
                
            if (epoch - best_val_epoch) * steps_per_epoch > args.stop_steps1:                
                break

        ema_model.load_state_dict(torch.load(best_ema_path))
        model1.load_state_dict(torch.load(best_model1_path))

        logger.info('model1 train finished')
        evaluate(validloader,ema_model,logger,'valid')    
        evaluate(testloader,ema_model,logger,'test')

        '''no_help_indices = np.concatenate((no_help_indices,confident_indices[current_num:]))
        confident_indices = confident_indices[:current_num]'''

        if len(confident_indices) >= len(all_indices):
            break

    #after all iterations, train a new model by all data

if __name__ == '__main__':
    main()