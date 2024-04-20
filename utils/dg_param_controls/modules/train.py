import os, sys
import numpy as np
import time
import torch

from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CURR_PATH)

sys.path.append(BASE_PATH)
from utils_dg import save_checkpoint, Summary, AverageMeter, ProgressMeter, accuracy
from metadata.indices_in_1k import indices_dict


def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, log_file):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        log_file,
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target, paths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #AugMix
        if args.exp_settings == 2:
            x_mix1, x_orig, x_mix2 = images
            bx = torch.cat((x_mix1, x_orig, x_mix2), 0).cuda(args.gpu, non_blocking=True)
        else:
            bx = images
        by = target.cuda(args.gpu, non_blocking=True)
        
        index_filter = indices_dict['imagenet-tin']

        logits = model(bx)[:, index_filter]
        
        #AugMix
        if args.exp_settings == 2:
            l_mix1, l_orig, l_mix2 = torch.split(logits, x_orig.size(0))
            loss = criterion(l_orig, by)

            p_orig, p_mix1, p_mix2 = F.softmax(l_orig, dim=1), F.softmax(l_mix1, dim=1), F.softmax(l_mix2, dim=1)

            M = torch.clamp((p_orig + p_mix1 + p_mix2) / 3., 1e-7, 1).log()
            loss += args.js_coefficient * (
                        F.kl_div(M, p_orig, reduction='batchmean') + F.kl_div(M, p_mix1, reduction='batchmean') +\
                        F.kl_div(M, p_mix2, reduction='batchmean')) / 3.

            output, target = l_orig, by
            images = x_orig
        else:
            loss = criterion(logits, by)
            output, target = logits, by
            # images = x_orig

        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        correct = accuracy(output, target, topk=(1,))
        correct = correct.reshape(-1)

        correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
        acc1 = correct_k.mul_(100.0 / images.size(0))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
