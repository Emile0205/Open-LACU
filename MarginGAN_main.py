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
import os
import math
import time
import wandb
import shutil
import logging
import argparse


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
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from MarginGAN_utils import *


LOG = logging.getLogger('main')
logging.basicConfig(filename='output.log', encoding='utf-8', level=logging.INFO)
logging.getLogger('PIL').setLevel(logging.WARNING)

args = None
best_prec1 = 0
global_step = 0


def main(context):
    global global_step
    global best_prec1

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    if not args.dataset == 'mnist':
        dataset_config = datasets.__dict__[args.dataset]()
        num_classes = dataset_config.pop('num_classes')
        num_classes = 6
        train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
    
        outdataset_config = datasets.__dict__[args.out_dataset]()
        out_loader = create_out_data_loaders(**outdataset_config, args=args)
        
    else:
        dataset_config = datasets.__dict__[args.dataset]()
        num_classes = dataset_config.pop('num_classes')
        train_loader, eval_loader = create_data_loaders(**dataset_config, args=args, mnist = True)
    
        print(sadsa)
        #outdataset_config = datasets.__dict__[args.out_dataset]()
        #out_loader = create_out_data_loaders(**outdataset_config, args=args)    
    
    
    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        model = model_factory(**model_params)

        if ema:  # # exponential moving average
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    G = generator(input_dim=args.z_dim, output_dim=3, input_size=32)
    D = discriminator(input_dim=3, output_dim=1, input_size=32)

    model.cuda()
    ema_model.cuda()
    G.cuda()
    D.cuda()

    LOG.info(parameters_string(model))

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
                                                                
    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

    BCEloss = nn.BCELoss().cuda()

    # optionally resume from a checkpoint
    if args.resume:
        LOG.info("=> loading checkpoint '{}'".format(args.resume))

        best_file = os.path.join(args.resume, 'best.ckpt')
        G_file = os.path.join(args.resume, 'G.pkl')
        C_file = os.path.join(args.resume, 'D.pkl')

        assert os.path.isfile(best_file), "=> no checkpoint found at '{}'".format(best_file)
        assert os.path.isfile(G_file), "=> no checkpoint found at '{}'".format(G_file)
        assert os.path.isfile(C_file), "=> no checkpoint found at '{}'".format(C_file)

        checkpoint = torch.load(best_file)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        G.load_state_dict(torch.load(G_file))
        D.load_state_dict(torch.load(C_file))

        LOG.info('----------------best_precl----------------', best_prec1)

        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True

    if args.evaluate:
        #image_frame_dim = int(np.floor(np.sqrt(128)))
        #generated_images_dir = 'generated_images/' + args.dataset
        #for i, ((input, ema_input), target) in enumerate(train_loader):
        #    samples = input.cpu().data.numpy().transpose(0, 2, 3, 1)
        #    
        #    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
        #                generated_images_dir + '/' + 'real.png')
        #    break
            
        #visualize_results(G, (180 + 1), args)
     
        LOG.info("\nEvaluating the model:")
        ema_prec1, ema_auroc, ema_oscr = validate(eval_loader, model, ema_model, ema_validation_log, global_step, 181, out_loader = out_loader, LOG = LOG, args = args)
        LOG.info("")
        print(asda)
        return

    #wandb.init(name='Margin-GAN', project='Comparing_GANs', config=args)
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, G, D, G_optimizer, D_optimizer, epoch, training_log, BCEloss)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            
            start_time = time.time()
            #LOG.info("Evaluating the primary model:")
            #prec1, auroc, oscr= validate(D, eval_loader, model, validation_log, global_step, epoch + 1, out_loader = out_loader, LOG = LOG, args = args)
            LOG.info("\nEvaluating the model:")
            ema_prec1, ema_auroc, ema_oscr = validate(eval_loader, model, ema_model, ema_validation_log, global_step, epoch + 1, out_loader = out_loader, LOG = LOG, args = args)
            LOG.info("")
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
            
            '''                
            wandb.log({"test/1.ssl_acc": prec1,
                       "test/2.ssl_acc_ema": ema_prec1,
                       "test/1.osr_auroc": auroc,
                       "test/2.osr_auroc_ema": ema_auroc,
                       "test/oscr": oscr,
                       "test/oscr_ema": ema_oscr})
            '''                                     
           
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)
            torch.save(G.state_dict(), os.path.join(checkpoint_path, 'G.pkl'))
            torch.save(D.state_dict(), os.path.join(checkpoint_path, 'D.pkl'))
            
            


def train(train_loader, model, ema_model, optimizer, G, D, G_optimizer, D_optimizer, epoch, log, BCEloss):
    global global_step

    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()
    D.train()
    G.train()

    end = time.time()

    for i, ((input, ema_input), target) in enumerate(train_loader):
        target[target >= 6] = -1
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader), args)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        

        input_var = torch.autograd.Variable(input.cuda())
        ema_input_var = torch.autograd.Variable(ema_input.cuda())
        target_var = torch.autograd.Variable(target.cuda())
                
        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)
        
        ema_logit, _, _ = ema_model(ema_input_var)
        logit1, logit2, real_features = model(input_var)

        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
        ema_class_loss = class_criterion(ema_logit, target_var) / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())        
               
        if args.logit_distance_cost >= 0:   #args.logit_distance_cost = 0.01
            class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.item())

        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0            

        class_loss = (class_criterion(class_logit, target_var)) / minibatch_size
        
        meters.update('class_loss', class_loss.item())
                

        if args.consistency: # args.consistency = 100
            consistency_weight = get_current_consistency_weight(epoch, args)
            meters.update('cons_weight', consistency_weight)                        
            consistency_loss = consistency_weight * (consistency_criterion(cons_logit, ema_logit) / minibatch_size)
            meters.update('cons_loss', consistency_loss.item())
            
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)        
        
        z_ = torch.rand((args.generated_batch_size, args.z_dim))
        z_ = z_.cuda()
        G_ = G(z_)
        c_fake_pred, _, fake_features = model(G_)

        #Margin-GAN inverse loss
        c_fake_pred_soft = F.softmax(c_fake_pred, dim=1)
        with torch.no_grad():
            c_fake_wei = torch.max(c_fake_pred_soft, 1)[1]
            c_fake_wei = c_fake_wei.view(-1, 1)
            c_fake_wei = torch.zeros(args.generated_batch_size, 6).cuda().scatter_(1, c_fake_wei, 1)
                    
        c_fake_loss = nll_loss_neg(c_fake_pred_soft, c_fake_wei)
        
        loss = class_loss + consistency_loss + res_loss + generated_weight(epoch) * c_fake_loss        
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 5))
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100. - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100. - prec5[0], labeled_minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit.data, target_var.data, topk=(1, 5))
        meters.update('ema_top1', ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], labeled_minibatch_size)
        meters.update('ema_top5', ema_prec5[0], labeled_minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5[0], labeled_minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            LOG.info('Epoch: {0} - {1}\{2} \t Time: {meters[data_time]:.3f} \t\t Prec@1 {meters[top1]:.3f}'.format(epoch, i, len(train_loader), meters = meters))

        # update D network
        temp_lab = (target != -1).nonzero(as_tuple = True)[0]
        temp_unlab = (target == -1).nonzero(as_tuple = True)[0]
        
        D_optimizer.zero_grad()

        D_real, _ = D(input_var)
        G_ = G(z_)
        D_fake, _ = D(G_)
        
        D_real_loss = BCEloss(D_real[temp_unlab], torch.ones_like(D_real[temp_unlab]))
        D_fake_loss = BCEloss(D_fake, torch.zeros_like(D_fake))
        #D_abc_loss = BCEloss(D_real[temp_lab], torch.zeros_like(D_real[temp_lab]))
        
        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        D_optimizer.step()
        
        
        # update G network
        G_optimizer.zero_grad()
        G_ = G(z_)
        
        D_fake, D_fake_features = D(G_)
        
        
        #Margin-GAN generator
        G_loss_D = BCEloss(D_fake, torch.ones_like(D_fake))
            
        C_fake_pred, _, _ = model(G_)
        C_fake_pred = F.log_softmax(C_fake_pred, dim=1)
        with torch.no_grad():
            C_fake_wei = torch.max(C_fake_pred, 1)[1]
        G_loss_C = F.nll_loss(C_fake_pred, C_fake_wei)

        G_loss = G_loss_D + generated_weight(epoch) * G_loss_C
        if epoch <= 10:
            G_loss_D.backward()
        else:
            G_loss_D.backward(retain_graph=True)
            G_loss_C.backward()

                
        G_optimizer.step()

        if i % args.print_freq == 0:
            LOG.info("\t\t\t D_loss: %.8f, \t\t G_loss: %.8f, \t\t C_loss: %.8f" %
                    (D_loss.item(), G_loss.item(), loss.item()))
            

    with torch.no_grad():
        visualize_results(G, (epoch + 1), args)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # #
    args = cli.parse_commandline_args()  # #
    main(RunContext(__file__, 0))  
    
    
    
    
    
'''
python3 MarginGAN_main.py     --dataset cifar10     --train-subdir train+val     --eval-subdir test     --batch-size 128     --labeled-batch-size 31     --arch cifar_shakeshake26     --consistency-type mse     --consistency-rampup 5     --consistency 100.0     --logit-distance-cost 0.01     --weight-decay 2e-4     --lr-rampup 0     --lr 0.05     --nesterov True     --labels data-local/labels/cifar10/4000_balanced_labels/00.txt      --epochs 181     --lr-rampdown-epochs 210     --ema-decay 0.97     --generated-batch-size 32     --out_dataset cifar100 

'''
    
    
    
    
    
    
    
    
    
    
