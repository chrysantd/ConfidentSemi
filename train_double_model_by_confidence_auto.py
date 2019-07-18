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
    parser.add_argument("--label-num",type=int,default=1000,
                        help="labeled data size") 
    parser.add_argument("--val-num",type=int,default=7320,
                        help="valid data size")                      
    parser.add_argument("--snapshot-dir", type=str, default='snapshots',
                        help="snapshot directory")
    parser.add_argument("--dataset", type=str, default='svhn',choices=['svhn','cifar10'],
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
    parser.add_argument("--initialize-from", type=str, required=False,
                        help="pretrained model for initialization") 
    parser.add_argument("--initialize-from2", type=str, required=False,
                        help="pretrained model for initialization") 
    parser.add_argument("--sample-method", type=str, default='max',choices=['max','sample'],
                        help="sample method")
    parser.add_argument("--drop-rate",type=float, default=0.1,
                        help="drop rate of training data")
    parser.add_argument("--base-ratio",type=float, default=0.1,
                        help="base ratio of confident data")    
    parser.add_argument("--round-ratio",type=float, default=0.1,
                        help="round ratio of confident data")
    parser.add_argument("--round-ratio2",type=float, default=0.02,
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

def get_chunks(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


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


def train(trainloader,model,optimizer,epoch,logger=None):
    model.train()
    len_iter = len(iter(trainloader))
    #global_step = epoch * len_iter
    
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

    
        #weight_cl = cal_consistency_weight(epoch, end_ep=(args.max_epoch//2), end_w=1.0)
        weight_cl = 1.0
        #global_step = global_step + 1

        train_loss = loss_ce + weight_cl * loss_cl

        train_loss.backward()
        
        optimizer.step()
        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item()))


def train_model1(confident_loader,unconfident_loader,model,optimizer,epoch,logger=None):
    model.train()

    conf_iter = iter(confident_loader)
    unconf_iter = iter(unconfident_loader)

    len_iter = max(len(conf_iter),len(unconf_iter)) * 2
    #global_step = epoch * len_iter

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
        
    
        #weight_cl = cal_consistency_weight(epoch, end_ep=(args.max_epoch//2), end_w=1.0)
        weight_cl = 1.0
        #global_step = global_step + 1

        train_loss = loss_ce + weight_cl * loss_cl

        optimizer.zero_grad()
        train_loss.backward()        
        optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item()))


def train_model1_mix(confident_loader,unconfident_loader,model,optimizer,epoch,iround,logger=None):
    model.train()

    conf_iter = iter(confident_loader)
    unconf_iter = iter(unconfident_loader)

    len_iter = max(len(conf_iter),len(unconf_iter)) * 2
    global_step = epoch * len_iter

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
        
        sc = input_conf.shape[0]
        su = input_unconf.shape[0]
        target_conf = target_conf.long()
        y_hot_conf = torch.zeros(sc, args.num_classes).scatter_(1, target_conf.view(-1,1), 1)
        
        if args.use_gpu:
            input_conf = input_conf.cuda()
            target_conf = target_conf.cuda()
            input1_conf = input1_conf.cuda()
            input_unconf = input_unconf.cuda()
            input1_unconf = input1_unconf.cuda() 
            y_hot_conf = y_hot_conf.cuda()       
                

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)

        idx = torch.randperm(sc)

        
        input_a, input_b = input_conf, input_conf[idx]
        input1_a, input1_b = input1_conf, input1_conf[idx]
        target_a, target_b = target_conf,target_conf[idx]
        mixed_input = l * input_a + (1 - l) * input_b
        mixed_input1 = l * input1_a + (l - 1) * input1_b
        
        input_all = torch.cat([mixed_input,input_unconf],0)
        input1_all = torch.cat([mixed_input1,input1_unconf],0)

        output_all = model(input_all)
        with torch.no_grad():
            output1_all = model(input1_all)

        loss_ce = l * ce_fn(output_all[:sc],target_a) + (1 - l) *ce_fn(output_all[:sc],target_b)
        loss_cl = mse_fn( torch.softmax(output_all[sc:],1), torch.softmax(output1_all[sc:],1) )
        #loss_unconf = l1_fn(torch.sum(torch.softmax(output1_all[sc:],1) ** 2 , 1), torch.cuda.FloatTensor(su).fill_(1)) / float(args.num_classes)
        
        #weight_cl_max = 20.0
        #weight_cl = cal_consistency_weight(global_step, end_ep=args.max_epoch * len_iter,init_w= weight_cl_max * iround /(args.round) ,end_w= weight_cl_max * (iround+1) /(args.round))
        #weight_cl = 3.0
        weight_cl = cal_consistency_weight(global_step, end_ep=args.max_epoch * len_iter,init_w= 0.0 ,end_w= 6.0)
        global_step = global_step + 1

        #train_loss = loss_ce + weight_cl * (loss_cl + loss_unconf)
        train_loss = loss_ce + weight_cl * loss_cl

        optimizer.zero_grad()
        train_loss.backward()        
        optimizer.step()

        if logger is not None:
            #logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}, loss_unconf = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item(),loss_unconf.item()))
            logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item()))


def train_model1_mix2(trainloader,model,optimizer,epoch,logger=None):
    model.train()
    #global_step = epoch * len_iter

    iter_loader = iter(trainloader)
    len_iter = max(512,len(iter_loader))
    
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


def train_with_vat(trainloader,model,optimizer,epoch,logger=None):
    model.train()
    len_iter = len(iter(trainloader))
    #global_step = epoch * len_iter
    
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
        
        loss_ce = ce_fn(y_pred,y)
    
        #weight_cl = cal_consistency_weight(epoch, end_ep=(args.max_epoch//2), end_w=1.0)
        weight_cl = 1.0
        #global_step = global_step + 1

        r_vadv = vat_utils.generate_virtual_adversarial_perturbation(
            x, y_pred, model, use_gpu=args.use_gpu
        )
        y_pred = model(x + r_vadv)
        distribution = F.log_softmax(y_pred, 1)
        distribution1 = F.softmax(y_pred1, 1)

        loss_cl = kl_fn(distribution, distribution1)

        train_loss = loss_ce + weight_cl * 0.3 * loss_cl

        train_loss.backward()
        
        optimizer.step()
        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item()))


def train_with_mix(trainloader,model,optimizer,epoch,max_epoch,logger=None):
    model.train()
    len_iter = len(iter(trainloader))
    global_step = epoch * len_iter
    
    for x , y, x1 in trainloader:

        size = x.size(0)
        y = y.long()
        y_hot = torch.zeros(size, args.num_classes).scatter_(1, y.view(-1,1), 1)

        if args.use_gpu:                
            x = x.cuda()
            y = y.cuda()
            x1 = x1.cuda()
            y_hot = y_hot.cuda()
        
        feat = model.feat(x)
        output = model.fc(feat)
        pred = torch.softmax(output,1)
        
        with torch.no_grad():
            feat1 = model.feat(x1)
            output1 = model.fc(feat1)
            pred1 = torch.softmax(output1,1)         


        #loss_ce = ce_fn(y_pred,y)

        loss_ce = ce_fn(output,y)
        loss_cl = mse_fn(feat,feat1)
        #loss_cl = mse_fn(pred,pred1)
        weight_cl = cal_consistency_weight(global_step, end_ep=max_epoch * len_iter // 2,init_w=0.0,end_w=5.0)

        global_step += 1

        train_loss = loss_ce +  weight_cl * loss_cl

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if logger is not None:
            logger.info('epoch{} : loss_ce = {:.4f} , loss_cl = {:4f}'.format(epoch, loss_ce.item(),loss_cl.item()))


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


def compute_confidences(model,dst1,dst2):
    model.eval()
    
    with torch.no_grad():
        loader1 = torch.utils.data.DataLoader(
            dst1,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True)

        loader2 = torch.utils.data.DataLoader(
            dst2,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            pin_memory=True
        )

        confidences = torch.FloatTensor(len(dst2))
        confidences.fill_(float('inf'))
    
        base_feats  = torch.FloatTensor()
        base_labels = torch.LongTensor()        
        base_probs = torch.FloatTensor()

        if args.use_gpu:
            base_feats = base_feats.cuda() 
            base_labels = base_labels.cuda()
            base_probs = base_probs.cuda()
            confidences = confidences.cuda()

        for x, y, x1 in loader1:
            size = x.size(0)
            y = y.long()
            if args.use_gpu:
                x = x.cuda()  
                y = y.cuda()   
                x1 = x1.cuda() 
            
            feats = model.feat(x)
            outputs = model.fc(feats)            
            outputs1 = model(x1)
            probs,labels = torch.max(torch.softmax(outputs,1),1)
            probs1,labels1 = torch.max(torch.softmax(outputs1,1),1)

            same_idx = (y == labels) & (y == labels1)
            
            probs = (probs + probs1) / 2

            base_feats = torch.cat([base_feats,feats[same_idx]],0)
            base_labels = torch.cat([base_labels,labels[same_idx]],0)
            base_probs = torch.cat([base_probs,probs[same_idx]],0)

        idx = 0       

        for x, y, x1 in loader2:
            y = y.long()
            if args.use_gpu:
                x = x.cuda()          
                y = y.cuda()  
                x1 = x1.cuda()

            feats = model.feat(x)
            feats1 = model.feat(x1)
            

            outputs = model.fc(feats)            
            outputs1 = model.fc(feats1)

            probs,labels  = torch.max(torch.softmax(outputs,1),1)
            probs1,labels1 = torch.max(torch.softmax(outputs1,1),1)

            labels = labels.long()
            labels1 = labels1.long()
            
            size = feats.size()[0]

            for i in range(size):
                if y[i] == labels[i] == labels1[i]:
                    confidences[idx+i] = min(compute_confidence(feats[i],labels[i],probs[i],base_feats,base_labels,base_probs), compute_confidence(feats1[i],labels1[i],probs1[i],base_feats,base_labels,base_probs))                

            idx +=  size
        
        return confidences.cpu()


def compute_confidence(feat,label,probs,base_feats,base_labels,base_probs):
    '''print(feat)
    print(label)
    print(base_feats)
    print(base_labels)'''
    confidences = ((label==base_labels).float() *(1-base_probs) *torch.sum((feat - base_feats) ** 2,1))
    #confidences = ((label==base_labels).float() *torch.sum((feat - base_feats) ** 2,1))
    return torch.min(confidences[confidences.nonzero()])


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())
    

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


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

    if not osp.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

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
    

    if args.initialize_from and osp.exists(args.initialize_from):
        model1.load_state_dict(torch.load(args.initialize_from))
    
    if args.initialize_from2 and osp.exists(args.initialize_from2):
        model2.load_state_dict(torch.load(args.initialize_from2))

    if args.use_gpu:
        model1 = model1.cuda()
        model2 = model2.cuda()
    
    
    
    optimizer1 = AdamW(model1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr, weight_decay=0)
    optimizer2 = AdamW(model2.parameters(), lr=args.lr, weight_decay=0)

    df = pd.DataFrame()
    stats_path = osp.join(args.snapshot_dir,'stats.txt')

    prop = args.base_ratio


    confident_indices = np.array([],dtype=np.int64)
    train_labels = np.array([],dtype=np.int32)

    _ , best_model1_acc = evaluate(validloader,model1,prefix='val')
    _ , best_model2_acc = evaluate(validloader,model2,prefix='val')
    best_model1_path = osp.join(args.snapshot_dir,'best_model1.pth')
    best_model2_path = osp.join(args.snapshot_dir,'best_model2.pth')
    torch.save(model1.state_dict(), best_model1_path)
    torch.save(model2.state_dict(), best_model2_path)
    skip_model2 = False
    end_iter = False


    for i_round in range(args.round):

        model1_save_path = osp.join(args.snapshot_dir,'model1_round_{}.pth'.format(i_round))
        model2_save_path = osp.join(args.snapshot_dir,'model2_round_{}.pth'.format(i_round))

        logger = logging.getLogger('model2_round_{}'.format(i_round))
        file_handler = logging.FileHandler(osp.join(args.snapshot_dir,'model2_round_{}.txt'.format(i_round)))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)          

        if not skip_model2:
            probs  = predict_probs(model1,unlabel_loader2)
            '''probs = np.zeros((len(unlabelset),args.num_classes))

            for i in range(args.K):
                probs  += predict_probs(model1,unlabel_loader)
            
            probs /= args.K'''

            samples = probs.argmax(axis=1).astype(np.int32)
            #samples2 = sample_by_probs(probs,1)[0].astype(np.int32)
            #samples[confident_indices] = train_labels[confident_indices]    
            #samples[unconfident_indices] = samples2[unconfident_indices]
            
            df2 = create_basic_stats_dataframe()
            df2['iter'] = i_round
            df2['train_acc'] = accuracy_score(unlabel_y,samples)   
            df = df.append(df2, ignore_index=True)
            df.to_csv(stats_path,index=False)


            #phase2: train model2            

            train_labels = samples

            unlabelset.train_labels_ul = train_labels

            trainloader = torch.utils.data.DataLoader(
                unlabelset,
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
                train_with_mix(trainloader,model2,optimizer2,epoch,args.max_epoch2,logger=logger)

                #torch.save(model2.state_dict(), model2_save_path)
                #evaluate(validloader,model2,logger,'val')

                if epoch >= args.min_epoch and args.evaluate_epoch> 0 and epoch % args.evaluate_epoch == 0:
                    val_loss,val_acc = evaluate(validloader,model2,logger,'val')

                    if args.eval_method == 'acc' and val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        best_val_loss = val_loss
                        best_val_epoch = epoch
                        torch.save(model2.state_dict(), model2_save_path)
                        evaluate(testloader,model2,logger,'test')
                            
                    elif args.eval_method == 'loss' and val_loss < best_val_loss:
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
        
        else:
            logger.info('skip!')

        
        #phase3: get confidence of unlabeled data by labeled data, split confident and unconfident data
        train_probs = np.zeros((len(unlabelset),args.num_classes))

        for i in range(args.K):
            train_probs  += predict_probs(model2,unlabel_loader)
        
        train_probs /= args.K
        
        train_labels = train_probs.argmax(axis=1).astype(np.int32)
        unlabelset.train_labels_ul = train_labels

        all_indices = np.arange(training_size).astype(np.int64)

        confidences = compute_confidences(model2,labelset,unlabelset)
        confident_indices  = np.array([],dtype=np.int64)
        for j in range(args.num_classes):
            k_size = min(int(training_size*prop)//args.num_classes, int((train_labels == j).sum()))
            logger.info('class: {}, confident size: {}'.format(j,k_size))
            j_idx = torch.LongTensor(all_indices[train_labels == j])            
            j_idx_top = confidences[j_idx].topk(k_size,largest=False)[1]
            confident_indices = np.concatenate( (confident_indices, j_idx[j_idx_top].data.numpy().astype(np.int64)))
        
        

        acc = accuracy_score(unlabel_y[confident_indices],train_labels[confident_indices])        
        logger.info('confident data prop: {:4f}, actual: {:4f}, acc: {:4f}'.format(prop, len(confident_indices)/len(unlabel_y), acc)) 

        all_unlabeled_indices = np.arange(training_size)
        mask = np.zeros(all_unlabeled_indices.shape,dtype=bool)
        mask[confident_indices] = True
        other_indices = all_unlabeled_indices[~mask]

        #unconfident_indices = confidences.topk(int(training_size*0.1))[1].data.numpy() 

        #use sudo and true labels separately
        confident_dataset = torch.utils.data.Subset(unlabelset,confident_indices)
        other_dataset = torch.utils.data.Subset(unlabelset,other_indices)


        #phase4: refine model1 by confident data and reward data
        train_dataset = torch.utils.data.ConcatDataset([confident_dataset,labelset])


        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size= args.batch_size//2,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True 
        )


        '''batch_sampler = TwoStreamBatchSampler(
            np.arange(len(confident_dataset)),
            np.arange(len(labelset)) + len(confident_dataset),
            args.batch_size//2,
            args.batch_size//4
        )        

        trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True 
        )'''
        
        other_loader = torch.utils.data.DataLoader(
            other_dataset,
            batch_size= args.batch_size//2,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True 
        )



        logger = logging.getLogger('model1_round_{}'.format(i_round))
        file_handler = logging.FileHandler(osp.join(args.snapshot_dir,'model1_round_{}.txt'.format(i_round)))
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
            #train_model1_mix(trainloader,other_loader,model1,optimizer1,epoch,i_round,logger)
            train_model1_mix2(trainloader,model1,optimizer1,epoch,logger)
            if epoch >= args.min_epoch and args.evaluate_epoch> 0 and epoch % args.evaluate_epoch == 0:
                val_loss,val_acc = evaluate(validloader,model1,logger,'valid')
                
                if args.eval_method == 'acc' and val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_val_epoch = epoch
                    torch.save(model1.state_dict(), model1_save_path)
                    evaluate(testloader,model1,logger,'test')    
                        
                elif args.eval_method == 'loss' and val_loss < best_val_loss:
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
            prop += args.round_ratio
            #model1.load_state_dict(torch.load(best_model1_path))
            #model2.load_state_dict(torch.load(best_model2_path))    
            #skip_model2 = True        
        else:         
            prop += args.round_ratio2 
            torch.save(model1.state_dict(), best_model1_path)
            best_model1_acc = best_val_acc
            #skip_model2 = False
        
        if end_iter:
            break

        if prop >= 1:
            prop = 1.0
            end_iter = True 

    #after all iterations, train a new model by all data


    

if __name__ == '__main__':
    main()