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
import vat_utils

import loader_svhn as svhn
import loader_cifar as cifar

from autoaugment import CIFAR10Policy, SVHNPolicy, Cutout
from adamw import AdamW

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
    parser.add_argument("--dataset", type=str, default='cifar10',choices=['svhn','cifar10'],
                        help="dataset type")
    parser.add_argument("--model", type=str, default='wide28',choices=['wide22','wide28','wide28_2'],
                        help="dataset type")     
    parser.add_argument('--boundary',default=0, type=int, help='different label/unlabel division [0,9]')
    parser.add_argument("--use-gpu",default=False,action='store_true', 
                        help="whether to use gpu")
    parser.add_argument("--vat",default=False,action='store_true', 
                        help="train_with_vat")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which gps to use")    
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--K', default=10, type=float)
 

    #model and loader settings
    parser.add_argument("--batch-size", type=int, default=512,
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
    parser.add_argument("--init2", type=str, required=False,
                        help="pretrained model for initialization") 
    parser.add_argument("--sample-method", type=str, default='max',choices=['max','sample'],
                        help="sample method")
    parser.add_argument("--drop-rate",type=float, default=0.1,
                        help="drop rate of training data")
    parser.add_argument("--base-ratio",type=float, default=0.1,
                        help="base ratio of confident data")    
    parser.add_argument("--round-ratio",type=float, default=0.2,
                        help="round ratio of confident data")
    #optimizer settings
    
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Base learning rate for training with polynomial decay.")    
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.001,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--min-epoch", type=int, default=0,
                        help="Max training epoch.")
    parser.add_argument("--max-epoch", type=int, default=100,
                        help="Max training epoch.")
    parser.add_argument("--max-epoch2", type=int, default=100,
                        help="Max training epoch.")
    parser.add_argument("--stop-epoch", type=int, default=0,
                        help="early stop epoch.")
    parser.add_argument("--evaluate-epoch", type=int, default=1,
                        help="epoch inteval for evaluation, set 0 to forbid")
    parser.add_argument("--eval-method", type=str, default='acc',choices=['loss','acc'],
                        help="epoch inteval for evaluation, set 0 to forbid")
    
    return parser.parse_args()

args = get_arguments()

ce_fn = torch.nn.CrossEntropyLoss()
ce_fn2 = torch.nn.CrossEntropyLoss(reduction=None)
mse_fn = torch.nn.MSELoss()
kl_fn = torch.nn.KLDivLoss(reduction='batchmean')
l1_fn = torch.nn.L1Loss()


def train_model1_mix(trainloader,model,model2,optimizer,epoch,logger=None):
    model.train()
    model2.eval()
    #global_step = epoch * len_iter

    iter_loader = iter(trainloader)
    len_iter = max(512,len(iter_loader))
    
    for i in range(len_iter):

        try:
            x , y, x1 = next(iter_loader)
        except StopIteration:
            iter_loader = iter(trainloader)
            x , y, x1 = next(iter_loader) 

        
        y = y.long()

        if args.use_gpu:                
            x = x.cuda()
            y = y.cuda()
            x1 = x1.cuda()    
    
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        x_concat = torch.cat([x,x1])
        y_concat = torch.cat([y,y])
        size = x_concat.size(0)
        idx = torch.randperm(size)  
        
        with torch.no_grad():
            '''weights = torch.max(torch.softmax(model2(x_concat),1),1)[0].view(-1,1)
            y_concat = torch.zeros(size, args.num_classes).cuda().scatter_(1, y_concat.view(-1,1), 1) * weights'''
            y_concat = torch.softmax(model2(x_concat),1)
        
        
        input_a, input_b = x_concat, x_concat[idx]
        target_a, target_b = y_concat, y_concat[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        mixed_output = model(mixed_input)

        #loss_ce = ce_fn(y_pred,y)

        loss = -torch.mean(torch.sum(F.log_softmax(mixed_output, dim=1) * mixed_target, dim=1))
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss = {:.4f}'.format(epoch, loss.item()))


def train_model2(trainloader,model,optimizer,epoch,max_epoch,logger=None):
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
    samples = np.zeros((round,size))

    for i in range(size):
        samples[:,i] = np.random.choice(range(classes),round,p=probs[i])

    samples = samples.astype(int)

    return samples


def create_basic_stats_dataframe():
    df = pd.DataFrame([[]])
    df['iter'] = 0
    df['train_acc'] = np.nan
    df['valid_loss'] = np.nan
    df['valid_acc'] = np.nan
    df['valid_epoch'] = 0
    return df

def adjust_learning_rate_adam(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""
    
    boundary = [args.max_epoch//3 * 2]
    lr = args.lr * 0.2 ** int(bisect.bisect_left(boundary, epoch))
    #print('Learning rate: %f'%lr)
    #print(epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def reset_learning_rate_adam(optimizer):
    """Sets the learning rate to the initial LR decayed by 5 at [240] epochs"""
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


def main():
    print(args)

    if not osp.exists(args.dir):
        os.makedirs(args.dir)

    if args.use_gpu:
        torch.cuda.set_device(args.gpu)
        cudnn.enabled = True 
        cudnn.benchmark = True


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

    labelset2 = dataloader(root=data_dir, split='label', download=True, 
        transform=transform_test, boundary=args.boundary, label_data_number = args.label_num, val_data_number = args.val_num)

    unlabelset = dataloader(root=data_dir, split='unlabel', download=True, 
        transform=transform_train, boundary=args.boundary, label_data_number = args.label_num, val_data_number = args.val_num)
    
    unlabelset2 = dataloader(root=data_dir, split='unlabel', download=True, 
        transform=transform_test, boundary=args.boundary, label_data_number = args.label_num, val_data_number = args.val_num)

    validset = dataloader(root=data_dir, split='valid', download=True, 
        transform=transform_test, boundary=args.boundary, label_data_number = args.label_num, val_data_number = args.val_num)

    testset = dataloader(root=data_dir, split='test', download=True, 
        transform=transform_test, label_data_number = args.label_num, val_data_number = args.val_num)

    unlabel_y =  np.array(unlabelset.train_labels_ul)

    training_size = unlabel_y.shape[0]

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
    

    if args.init and osp.exists(args.init):
        model1.load_state_dict(torch.load(args.init))
    
    if args.init2 and osp.exists(args.init2):
        model2.load_state_dict(torch.load(args.init2))

    if args.use_gpu:
        model1 = model1.cuda()
        model2 = model2.cuda()
    
    
    
    optimizer1 = AdamW(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=0)
    optimizer2 = AdamW(model2.parameters(), lr=args.lr, weight_decay=0)

    df = pd.DataFrame()
    stats_path = osp.join(args.dir,'stats.txt')

    prop = args.base_ratio

    _ , best_model1_acc = evaluate(validloader,model1,prefix='val')
    _ , best_model2_acc = evaluate(validloader,model2,prefix='val')
    best_model1_path = osp.join(args.dir,'best_model1.pth')
    best_model2_path = osp.join(args.dir,'best_model2.pth')
    torch.save(model1.state_dict(), best_model1_path)
    torch.save(model2.state_dict(), best_model2_path)
    skip_model2 = False
    end_iter = False

    round_ratio = args.round_ratio


    for i_round in range(args.round):

        model1_save_path = osp.join(args.dir,'model1_round_{}.pth'.format(i_round))
        model2_save_path = osp.join(args.dir,'model2_round_{}.pth'.format(i_round))

        logger = logging.getLogger('model2_round_{}'.format(i_round))
        file_handler = logging.FileHandler(osp.join(args.dir,'model2_round_{}.txt'.format(i_round)))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)          

        
        probs  = predict_probs(model1,unlabel_loader2)
        samples = probs.argmax(axis=1).astype(np.int32)
        
        df2 = create_basic_stats_dataframe()
        df2['iter'] = i_round
        df2['train_acc'] = accuracy_score(unlabel_y,samples)   
        df = df.append(df2, ignore_index=True)
        df.to_csv(stats_path,index=False)

        #phase2: train model2
        unlabelset.train_labels_ul = samples.copy()
        unlabelset2.train_labels_ul = samples.copy()

        trainset = ConcatDataset([labelset,unlabelset])
        trainset2 = ConcatDataset([labelset2,unlabelset2])

        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)
        
        trainloader2 = torch.utils.data.DataLoader(
            trainset2,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True)

        best_val_epoch = 0 
        best_val_loss = float('inf')
        best_val_acc = .0
        evaluate(validloader,model2,logger,'val')
        evaluate(testloader,model2,logger,'test')
        
        
        #reset_learning_rate_adam(optimizer2)
        for epoch in range(args.max_epoch2):
            #adjust_learning_rate_adam(optimizer2,epoch)            
            #train_with_vat(trainloader,model2,optimizer2,epoch,logger)
            if epoch % 10 > 0:
                train_model2(trainloader,model2,optimizer2,epoch,args.max_epoch2,logger=logger)
            else:
                train_model2(trainloader2,model2,optimizer2,epoch,args.max_epoch2,logger=logger)

            #torch.save(model2.state_dict(), model2_save_path)
            #evaluate(validloader,model2,logger,'val')

            if epoch >= args.min_epoch and args.evaluate_epoch> 0 and epoch % args.evaluate_epoch == 0:
                val_loss,val_acc = evaluate(validloader,model2,logger,'val')

                if val_acc >= best_val_acc:
                #if val_loss < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    torch.save(model2.state_dict(), model2_save_path)
                    evaluate(testloader,model2,logger,'test')

        model2.load_state_dict(torch.load(model2_save_path))        

        df.loc[df['iter'] == i_round,'valid_acc'] = best_val_acc
        df.loc[df['iter'] == i_round,'valid_loss'] = best_val_loss
        df.loc[df['iter'] == i_round,'valid_epoch'] = best_val_epoch
        df.to_csv(stats_path,index=False)

        logger.info('model2 train finished')        
        evaluate(validloader,model2,logger,'val')
        evaluate(label_loader2,model2,logger,'reward')        
        evaluate(testloader,model2,logger,'test')

        if best_val_acc > best_model2_acc:
            torch.save(model2.state_dict(), best_model2_path)            
                
        #phase3: get confidence of unlabeled data by labeled data, split confident and unconfident data
        probs  = predict_probs(model2,unlabel_loader2)
        samples = probs.argmax(axis=1).astype(np.int32)
        confidences = probs.max(axis=1)
        
        confident_indices  = np.array([],dtype=np.int64)
        all_indices = np.arange(training_size).astype(np.int64)

        for j in range(args.num_classes):
            k_size = min(int(training_size*prop)//args.num_classes, int((samples==j).sum()))
            logger.info('class: {}, confident size: {}'.format(j,k_size))
            j_idx_top = all_indices[samples==j][confidences[samples==j].argsort()[-k_size:]]
            confident_indices = np.concatenate( (confident_indices, j_idx_top))

        acc = accuracy_score(unlabel_y[confident_indices],samples[confident_indices])        
        logger.info('confident data prop: {:4f}, actual: {:4f}, acc: {:4f}'.format(prop, len(confident_indices)/len(unlabel_y), acc))

        all_unlabeled_indices = np.arange(training_size)
        mask = np.zeros(all_unlabeled_indices.shape,dtype=bool)
        mask[confident_indices] = True
        other_indices = all_unlabeled_indices[~mask]

        #unconfident_indices = confidences.topk(int(training_size*0.1))[1].data.numpy() 


        unlabelset.train_labels_ul = samples.copy()
        unlabelset2.train_labels_ul = samples.copy()

        #use sudo and true labels separately
        confident_dataset = torch.utils.data.Subset(unlabelset,confident_indices)
        confident_dataset2 = torch.utils.data.Subset(unlabelset2,confident_indices)


        #phase4: refine model1 by confident data and reward data
        train_dataset = torch.utils.data.ConcatDataset([confident_dataset,labelset])
        train_dataset2 = torch.utils.data.ConcatDataset([confident_dataset2,labelset2])


        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size= args.batch_size//2,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True 
        )
        
        trainloader2 = torch.utils.data.DataLoader(
            train_dataset2,
            batch_size= args.batch_size//2,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True 
        )



        logger = logging.getLogger('model1_round_{}'.format(i_round))
        file_handler = logging.FileHandler(osp.join(args.dir,'model1_round_{}.txt'.format(i_round)))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

        best_val_epoch = 0 
        best_val_loss = float('inf')
        best_val_acc = .0
        evaluate(validloader,model1,logger,'valid') 
        evaluate(testloader,model1,logger,'test')

        #reset_learning_rate_adam(optimizer1)
        for epoch in range(args.max_epoch):
            #adjust_learning_rate_adam(optimizer1,epoch)
            if epoch % 10 > 0:
                train_model1_mix(trainloader,model1,model2,optimizer1,epoch,logger)
            else:
                train_model1_mix(trainloader2,model1,model2,optimizer1,epoch,logger)
            if epoch >= args.min_epoch and args.evaluate_epoch> 0 and epoch % args.evaluate_epoch == 0:
                val_loss,val_acc = evaluate(validloader,model1,logger,'valid')
                
                if val_acc >= best_val_acc:
                #if val_loss < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    torch.save(model1.state_dict(), model1_save_path)
                    evaluate(testloader,model1,logger,'test')
                
        
        model1.load_state_dict(torch.load(model1_save_path))
                
        logger.info('model1 train finished')
        evaluate(validloader,model1,logger,'valid')    
        evaluate(testloader,model1,logger,'test')        

        if best_model1_acc >= best_val_acc:
            round_ratio = round_ratio * 2            
            #model1.load_state_dict(torch.load(best_model1_path))
            #model2.load_state_dict(torch.load(best_model2_path))    
            #skip_model2 = True        
        else:         
            round_ratio = max(round_ratio/1.5, args.round_ratio)
            torch.save(model1.state_dict(), best_model1_path)
            best_model1_acc = best_val_acc
            #skip_model2 = False
        
        prop += round_ratio
        
        if end_iter:
            break

        if prop >= 1:
            prop = 1.0
            end_iter = True 

    #after all iterations, train a new model by all data


    

if __name__ == '__main__':
    main()