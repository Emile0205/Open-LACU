# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# Thank the authors of mean teacher.
# The github address is https://github.com/CuriousAI/mean-teacher
# Our code is widely adapted from their repositories.

import re
import argparse
import os
import shutil
import time
import math
import logging


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from torchmetrics.image.fid import FrechetInceptionDistance
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *



def nll_loss_neg(y_pred, y_true):  # # #
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))

# batch_size * input_dim => batch_size * output_dim * input_size * input_size
class generator(nn.Module):  # # #
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 4,2,1可以将大小扩大一倍
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

# batch_size * input_dim * input_size * input_size => batch_size * output_dim
class discriminator(nn.Module):  # # #
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1), # 4，2，1缩小1倍
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.temp1 = nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024)
        self.temp2 = nn.BatchNorm1d(1024)
        self.temp3 = nn.LeakyReLU(0.2)
        self.temp4 = nn.Linear(1024, self.output_dim)
        self.temp5 = nn.Sigmoid()
        
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.temp1(x)
        x = self.temp2(x)
        x = self.temp3(x)
        features = x
        x = self.temp4(x)
        logits = self.temp5(x)

        return logits, features


def create_data_loaders(train_transformation,
                        eval_transformation,
                        args,
                        datadir = '',
                        mnist = False):
    
    if mnist == False:                    
        traindir = os.path.join(datadir, args.train_subdir)
        evaldir = os.path.join(datadir, args.eval_subdir)

        assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

        dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)
        
        testset = torchvision.datasets.ImageFolder(evaldir, eval_transformation)
        
    else:       
        dataset = torchvision.datasets.MNIST(root='./data/', train=True, download=False, transform=train_transformation)
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.MNIST(root='./data/', train=False, download=False, transform=eval_transformation)
        #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        

    if args.labels:
        with open(args.labels) as f:
            labels = dict(line.split(' ') for line in f.read().splitlines())
        labeled_idxs, unlabeled_idxs = data.relabel_dataset(dataset, labels)

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler)

    eval_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    return train_loader, eval_loader
    
    
    
def create_out_data_loaders(eval_transformation,
                            datadir, num_classes,
                            args):
                        
    evaldir = os.path.join(datadir, args.eval_subdir)

    eval_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False)

    return eval_loader
    


   


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha = 1 - alpha)


def visualize_results(G, epoch, args):
    G.eval()
    generated_images_dir = 'generated_images/' + args.dataset
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

    tot_num_samples = 64
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    sample_z_ = torch.rand((tot_num_samples, args.z_dim))

    sample_z_ = sample_z_.cuda()

    samples = G(sample_z_)


    samples = samples.mul(0.5).add(0.5)

    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)


    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      generated_images_dir + '/' + 'epoch%03d' % epoch + '.png')


def validate(eval_loader, model, ema_model, log, global_step, epoch, out_loader = None, LOG = None, args = None):
    meters = AverageMeterSet()
    _pred_ema_k, _pred_ema_u = [], []
    _pred_k, _pred_u = [], []
        
    torch.cuda.empty_cache()
    # switch to evaluate mode
    model.eval()
    ema_model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(eval_loader):
            target[target >= 6] = -1
            
            meters.update('data_time', time.time() - end)

            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            # compute output
            ema_class_output, _, _ = ema_model(input_var)
            class_output, _, _ = model(input_var)

            _pred_ema_k.append(ema_class_output[target != -1].data.cpu().numpy())
            #_pred_ema_u.append(ema_class_output[target == -1].data.cpu().numpy())
            
            _pred_k.append(class_output[target != -1].data.cpu().numpy())
            _pred_u.append(class_output[target == -1].data.cpu().numpy())
                 
            # measure accuracy and record loss
            prec1, prec5 = accuracy(ema_class_output.data, target_var.data, topk=(1, 5))
            meters.update('top1', prec1[0], labeled_minibatch_size)
            meters.update('top5', prec5[0], labeled_minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()
                                
                
    with torch.no_grad():             
        for batch_idx, (data, labels) in enumerate(out_loader):
            input_var = torch.autograd.Variable(data.cuda())
            
            with torch.set_grad_enabled(False):               
                ema_class_output, _, _ = ema_model(input_var)
                _pred_ema_u.append(ema_class_output.data.cpu().numpy())  
                
                #class_output, _, _ = model(input_var)
                #_pred_u.append(class_output.data.cpu().numpy())              
    
            
    LOG.info('')    
    LOG.info('Test set: \t\t Prec@1 {top1.avg:.3f}\t\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))   
    
            
    _pred_ema_k = np.concatenate(_pred_ema_k, 0)
    _pred_ema_u = np.concatenate(_pred_ema_u, 0)
    
    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
        
    # Out-of-Distribution detction evaluation
    LOG.info("Observed novel category detection using primary classifier")
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = metric_ood(x1, x2, LOG = LOG)['Bas']
    
    LOG.info("Unobserved novel category detection using exponentially moving average classifier")
    x1, x2 = np.max(_pred_ema_k, axis=1), np.max(_pred_ema_u, axis=1)
    results = metric_ood(x1, x2, LOG = LOG)['Bas']
    
    # OSCR
    #_oscr_socre = compute_oscr(_pred_ema_k, _pred_u, _labels)

    results['ACC'] = meters['top1'].avg
    results['OSCR'] = 0.0 * 100.

    LOG.info('OSCR: ' + str(results['OSCR'] ))

    return meters['top1'].avg, results['AUROC'], results['OSCR']

   


def save_checkpoint(state, is_best, dirpath, epoch):
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch, args):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def generated_weight(epoch):
    alpha = 0.0
    T1 = 10
    T2 = 60
    af = 0.3
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size.float()))
    return res
    
    
    
#############################################################################################################################
def get_curve_online(known, novel, stypes = ['Bas'], margins = False):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    
    for stype in stypes:
        known.sort()
        novel.sort()
        
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        
        num_k = known.shape[0]
        num_n = novel.shape[0]
        
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
                
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
                
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                    
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]
                    
                    
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
        
    return tp, fp, tnr_at_tpr95
    

def metric_ood(x1, x2, stypes = ['Bas'], verbose=True, LOG = None, margins = False):
    tp, fp, tnr_at_tpr95 = get_curve_online(x1, x2, stypes, margins)
    
    results = dict()
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    
    if margins:
        LOG.info('')
        LOG.info('Margins')
        LOG.info('       AUROC')
        
    else:
        LOG.info('       TNR \t AUROC')
        
    for stype in stypes:
        #if verbose:
        #    LOG.info('{stype:5s} '.format(stype=stype))
        results[stype] = dict()
        
        # TNR
        mtype = 'TNR'
        results[stype][mtype] = 100.*tnr_at_tpr95[stype]
        #if verbose:
        #    LOG.info(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype]/tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype]/fp[stype][0], [0.]])
        results[stype][mtype] = 100.*(-np.trapz(1.-fpr, tpr))
        #if verbose:
        #    LOG.info(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # DTACC
        mtype = 'DTACC'
        results[stype][mtype] = 100.*(.5 * (tp[stype]/tp[stype][0] + 1.-fp[stype]/fp[stype][0]).max())
        #if verbose:
        #    LOG.info(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # AUIN
        mtype = 'AUIN'
        denom = tp[stype]+fp[stype]
        denom[denom == 0.] = -1.
        pin_ind = np.concatenate([[True], denom > 0., [True]])
        pin = np.concatenate([[.5], tp[stype]/denom, [0.]])
        results[stype][mtype] = 100.*(-np.trapz(pin[pin_ind], tpr[pin_ind]))
        #if verbose:
        #    LOG.info(' {val:6.3f}'.format(val=results[stype][mtype]))
        
        # AUOUT
        mtype = 'AUOUT'
        denom = tp[stype][0]-tp[stype]+fp[stype][0]-fp[stype]
        denom[denom == 0.] = -1.
        pout_ind = np.concatenate([[True], denom > 0., [True]])
        pout = np.concatenate([[0.], (fp[stype][0]-fp[stype])/denom, [.5]])
        results[stype][mtype] = 100.*(np.trapz(pout[pout_ind], 1.-fpr[pout_ind]))
        #if verbose:
        #    LOG.info(' {val:6.3f}'.format(val=results[stype][mtype]))
        #    LOG.info('')
        
        if margins == True:
            LOG.info('      {val2:6.3f}'.format(val2 = results[stype]['AUROC']))
            LOG.info('')
            
        else:
            LOG.info('      {val1:6.3f} \t {val2:6.3f}'.format(val1 = results[stype]['TNR'], val2 = results[stype]['AUROC']))
        
    
    return results

def compute_oscr(pred_k, pred_u, labels):
    x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    pred = np.argmax(pred_k, axis=1)
    correct = (pred == labels)    
    
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values
    
    CCR = [0 for x in range(n+2)]
    FPR = [0 for x in range(n+2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n-1):
        CC = s_k_target[k+1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n+1] = 1.0
    FPR[n+1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n+1):
        h =   ROC[j][0] - ROC[j+1][0]
        w =  (ROC[j][1] + ROC[j+1][1]) / 2.0

        OSCR = OSCR + h*w

    return OSCR   
    
    
    


    
    
    
    
    
    
    
    
    
    

