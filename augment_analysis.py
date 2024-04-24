# Source : https://github.com/hendrycks/imagenet-r/issues/1
import argparse
import os
import random
import shutil
import time
import warnings
import math
import numpy as np
from PIL import ImageOps, Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from utils.dg_param_controls.modules.custom_datasets import DatasetFolder_withpath
from torchvision.datasets.folder import default_loader
import pickle

from utils.dg_param_controls.modules.datasets import dataset_path, get_dataset, get_train_dataset
from utils.dg_param_controls.utils_dg import log_string, count_parameters, save_checkpoint
from utils.dg_param_controls.modules.train import train
from utils.dg_param_controls.modules.validate import validate

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='AugMix ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=284, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# AugMix parameters
parser.add_argument('--js-coefficient', default=12., type=float, help='Jensen-Shannon loss scale parameter (lambda)')
# parser.add_argument('--aug-severity', default=1, type=int, help='aug severity')
parser.add_argument('--exp-settings', default=0, type=int, help='Experiment settings. \
                                                0: Basic aug only,\
                                                1: Colorjitter + PhotometricDistortion,\
                                                2: AugMix & DeepAugment')
parser.add_argument('--use-es-training', action='store_true', help='Use as data for training')
parser.add_argument('--data_root', default='~/datasets', type=str,
                    help='Datset root directory')


args = parser.parse_args()


best_acc1, best_acc1_c, best_acc1_es = 0, 0, 0
os.makedirs('aug_logs', exist_ok=True)
os.makedirs('results_dg_param_control', exist_ok=True)
log_file = f'aug_logs/aug_experiments_{args.exp_settings}_{int(args.use_es_training)}.txt'
LOG_FOUT = open(log_file, 'a+')
result_file = f'results_dg_param_control/aug_experiments.txt'
if not os.path.exists(result_file):
    RESULT_FOUT = open(result_file, 'a+')
    RESULT_FOUT.write('Id,Augmentations,Use-ES-data,ImageNet val., ImageNet-C val., ImageNet-ES val.\n')
else: 
    RESULT_FOUT = open(result_file, 'a+')


os.makedirs('ckpt', exist_ok=True)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    LOG_FOUT.close()


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_acc1_c, best_acc1_es
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model

    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    # checkpoint = torch.load('./augmix_init.pth.tar', map_location=lambda storage, loc: storage)
    # state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    # model.load_state_dict(state_dict)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    # optionally resume from a checkpoint
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            print('Start epoch:', args.start_epoch)
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # Training dataset settings: differs by the experiment settings.
    train_dataset = get_train_dataset(args.data_root, args.exp_settings, args.use_es_training)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    #### Validation datasets: ImageNet, ImageNet-C, ImageNet-ES
    im_dict = get_dataset('imagenet-tin', args.data_root, args.arch)
    im_c_dict = get_dataset('imagenet-c-tin', args.data_root, args.arch)
    im_es_dict = get_dataset('imagenet-es', args.data_root, args.arch)

    val_dataset_im = torch.utils.data.ConcatDataset(im_dict.values())
    val_dataset_im_c = torch.utils.data.ConcatDataset(im_c_dict.values())
    val_dataset_im_es = torch.utils.data.ConcatDataset(im_es_dict.values())

    val_loader_im = torch.utils.data.DataLoader(val_dataset_im,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader_im_c = torch.utils.data.DataLoader(val_dataset_im_c,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader_im_es = torch.utils.data.DataLoader(val_dataset_im_es,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)



    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / (args.lr * args.batch_size / 256.)))
    scheduler.step(args.start_epoch * len(train_loader))

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return


    log_string(LOG_FOUT, f'*'*50 + '\n')
    log_string(LOG_FOUT, f'Training starts... Experiments:{args.exp_settings}, Use ES data:{args.use_es_training}\n')

    for epoch in range(args.start_epoch, args.epochs):
        log_string(LOG_FOUT, f'===== Epoch {epoch} =====\n')
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        # train(train_loader, model, criterion, optimizer, scheduler, epoch, args, LOG_FOUT)

        # evaluate on validation set
        log_string(LOG_FOUT, '-'*10 + 'Imagenet validation' + '-'*10)
        acc1 = validate(val_loader_im, model, criterion, args, 'imagenet-tin', LOG_FOUT)

        log_string(LOG_FOUT, '-'*10 + 'Imagenet-C validation' + '-'*10)
        acc1_c = validate(val_loader_im_c, model, criterion, args, 'imagenet-c-tin', LOG_FOUT)
        
        log_string(LOG_FOUT, '-'*10 + 'Imagenet-ES validation' + '-'*10)
        acc1_es = validate(val_loader_im_es, model, criterion, args, 'imagenet-es', LOG_FOUT)
        

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc1_c, best_acc1_es = acc1_c, acc1_es
        log_string(LOG_FOUT, f'*** Best accuracy (Im, Im-C, Im-ES): {best_acc1}, {best_acc1_c}, {best_acc1_es}')

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, f'ckpt/aug_exp_{args.exp_settings}_{int(args.use_es_training)}.pt')

    exp_desc_map = {0:'Comp Aug.', 1:'Basic Digital Aug.', 2:'Advanced Digital Aug.'}
    log_string(RESULT_FOUT, f'{1 + args.use_es_training*3 + args.exp_settings}, {exp_desc_map[args.exp_settings]},' +
                        f'{args.use_es_training},{best_acc1}, {best_acc1_c}, {best_acc1_es}')
    
    LOG_FOUT.close()
    RESULT_FOUT.close()

    

if __name__ == '__main__':
    main()