import numpy as np
import torch
import torch.nn.functional as F
import math
import wandb
import datetime

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
    l1 = F.one_hot(y_a, num_classes)
    l2 = F.one_hot(y_b, num_classes)
    return lam * l1 + (1 - lam) * l2


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


def mixup_data_custom(x, y, y_bools, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, pairs of bools, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    y_bools_a, y_bools_b = y_bools, y_bools[index]

    # Mixing the boolean values
    mixed_y_bools = lam * y_bools_a + (1 - lam) * y_bools_b

    return mixed_x, y_a, y_b, mixed_y_bools, lam



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
        
def init_wandb(args):
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_project,
        name=args.desc,
        config=args,
    )
    wandb.run.save()
    return wandb.config
        
        

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
            
            
def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))



class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]