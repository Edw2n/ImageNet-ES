import os
import numpy as np
from enum import Enum
import torch

def log_string(log_f, out_str):
    log_f.write(out_str+'\n')
    log_f.flush()
    print(out_str)  

def count_parameters(model):
    cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return cnt
    # print("The number of parameters:", cnt)

def parse_acc_imagenet_c(acc_by_param, log_file, result_file, write_header=False):
    errs = {}
    for k, v in acc_by_param.items():
        dataset_name, arch, d_name, s = k.split('_')
        if d_name in errs:
            errs[d_name] += 100.0 - v            
        else:
            errs[d_name] = [100.0 -v]

    all_ces = []

    if write_header:
        log_string(result_file, 'Distortion,Severity,Error') # Header    

    for k, v in errs.items():
        dataset_name, arch, d, s = k.split('__')
        log_string(result_file, f'{d},{s},{v}')
        all_ces += v


    # log_string(log_file, f"===== Average CE by distortion =====")
    for k, v in errs.items():
        log_string(result_file, f"{k}, {np.mean(v)}")

    log_string(log_file, "*"*10 + f"Imagenet-c average of corruption error: {np.mean(all_ces)}" + "*"*10)


def parse_acc_imagenet_es(acc_by_param, log_file, result_file, write_header=False):
    
    if write_header:
        log_string(result_file, 'Light,Camera Parameter,Acc.') # Header
        
    for k, v in acc_by_param.items():
        dataset_name, arch, l, param = k.split('__')
        log_string(result_file, f'{l},{param},{v}')

    log_string(log_file, "*"*10 + f"Imagenet-ES Average acc: {np.mean(list(acc_by_param.values()))}" + "*"*10)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, log_file, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_file = log_file

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [self.prefix + " summary"]
        entries += [meter.summary() for meter in self.meters]
        # print(' '.join(entries))
        log_string(self.log_file, ' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        return correct

def collect_features(args):
    from collections import defaultdict

    dict1, dict2, dict3 = {}, {}, {}
    for d in os.listdir('features'):
        if os.path.isdir(os.path.join('features',d)):
            dataset_name, model, light, param_no = d.split('__')
            if dataset_name != args.dataset: continue
            key1 = 'param-control' if dataset_name == 'imagenet_es' else 'auto-exposure'
            key2 = light
            key3 = param_no
            
            dict4 = defaultdict(list)
            for f in os.listdir(os.path.join('features', d)):
                key4 = f.split('_')[0]
                feats = np.load(os.path.join('features', d,f))
                dict4[key4].append(feats)
                os.remove(os.path.join('features',d,f))
            dict3[key3] = dict4
            dict2[key2] = dict3
            dict1[key1] = dict2
            os.rmdir(os.path.join('features',d))

    torch.save(dict1, f'features/fvs-{args.arch}-{args.dataset}.pt')
