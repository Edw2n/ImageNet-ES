import os, sys
import numpy as np
import time
import torch
import torch.nn as nn

from torchvision.models.feature_extraction import create_feature_extractor

CURR_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CURR_PATH)

sys.path.append(BASE_PATH)
from utils import save_checkpoint, Summary, AverageMeter, ProgressMeter, accuracy
from metadata.indices_in_1k import indices_dict

def validate(val_loader, model, criterion, args, dataset_name, log_file, desc='ImageNet'):

    def run_validate(loader, detail_log, dataset_name, save_features, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target, paths) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    images = images.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                index_filter = indices_dict[dataset_name]
                is_timm = True if hasattr(args, 'timm') and args.timm else False
                arch = None if not hasattr(args, 'arch') else args.arch

                if is_timm and arch == 'res50': # Fine-tuned resnet 50 model weights
                    output = model(images)
                else:
                    if save_features:
                        dataset_name, arch, l, param = desc.split('__')
                        if is_timm:
                            feats = model.forward_features(images)
                        else:
                            return_nodes = {'layer4': 'layer4'}
                            feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
                            feats = feature_extractor(images)['layer4']

                        feats = nn.AdaptiveAvgPool2d((1, 1))(feats)
                        feats = torch.flatten(feats, 1).cpu().numpy()

                        for f, p in zip(feats, paths):
                            np.save( os.path.join('features',desc,f'{p.split("/")[-2]}_{os.path.basename(p)}.npy'), f)
                    
                    output = model(images)[:,index_filter]                
                
                loss = criterion(output, target)
                
                correct = accuracy(output, target, topk=(1,))
                correct = correct.reshape(-1)

                if detail_log is not None:
                    for c, p in zip(correct, paths):
                        detail_log.write(f'{p.split("/")[-2]},{os.path.basename(p)},{c*1}\n')
                

                correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
                acc1 = correct_k.mul_(100.0 / images.size(0))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                # top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    # top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1],
        log_file,
        prefix=f'{desc}: ')

    # switch to evaluate mode
    model.eval()
    detail_log = None
    save_features = False

    if hasattr(args, 'save_features') and args.save_features:
        if dataset_name in ['imagenet-es', 'imagenet-es-auto']:
            os.makedirs(os.path.join('features', desc), exist_ok=True)
            save_features = True
        else:
            print("WARNING: Only saving features for ImageNet-ES is supported!")
    
    if hasattr(args, 'save_details') and args.save_details:
        detail_log = open(os.path.join('logs', 'details', f'eval_details_{desc}'), 'w+')

    run_validate(val_loader, detail_log, dataset_name, save_features)
    if detail_log is not None:
        detail_log.close()

    if args.distributed:
        top1.all_reduce()
        # top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg