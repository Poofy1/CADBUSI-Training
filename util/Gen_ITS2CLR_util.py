import numpy as np
import torch
import torch.nn.functional as F
import math

class Args:
    def __init__(self, warm, start_epoch, warm_epochs, learning_rate, lr_decay_rate, num_classes, epochs, warmup_from, cosine, lr_decay_epochs, mix, mix_alpha, KD_temp, KD_alpha, teacher_path, teacher_ckpt):
        self.warm = warm
        self.start_epoch = start_epoch
        self.warm_epochs = warm_epochs
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.num_classes = num_classes
        self.epochs = epochs
        self.warmup_from = warmup_from
        self.cosine = cosine
        self.lr_decay_epochs = lr_decay_epochs
        self.mix = mix
        self.mix_alpha = mix_alpha
        self.KD_temp = KD_temp
        self.KD_alpha = KD_alpha
        self.teacher_path = teacher_path
        self.teacher_ckpt = teacher_ckpt

def mix_fn(x, y, alpha, kind):
    if kind == 'mixup':
        return mixup_data(x, y, alpha)
    elif kind == 'cutmix':
        return cutmix_data(x, y, alpha)
    elif kind == 'mixup_cutmix':
        if np.random.rand(1)[0] > 0.5:
            return mixup_data(x, y, alpha)
        else:
            return cutmix_data(x, y, alpha)
    else:
        raise ValueError()


def mix_target(y_a, y_b, lam, num_classes):
    # Check if inputs are already one-hot encoded
    if y_a.dim() > 1 and y_a.size(1) == num_classes:
        return lam * y_a + (1 - lam) * y_b
    else:
        # Convert to one-hot if they're indices
        l1 = F.one_hot(y_a.long(), num_classes)
        l2 = F.one_hot(y_b.long(), num_classes)
        return lam * l1 + (1 - lam) * l2



def prediction_anchor_scheduler(current_epoch, config, warmup_epochs = 0):
    if current_epoch < warmup_epochs:
        return config['initial_ratio']
    else:
        return config['initial_ratio'] + (config['final_ratio'] - config['initial_ratio']) * (current_epoch - warmup_epochs) / (config['total_epochs'] - warmup_epochs)
    

'''
modified from https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
'''
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



'''
modified from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
'''
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    bsz = x.size()[0]
    index = torch.randperm(bsz, device=x.device)
    
    bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
    mixed_x = x.detach().clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
        
        

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        warmup_to = eta_min + (args.learning_rate - eta_min) * (
            1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2

        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
