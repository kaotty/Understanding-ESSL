import hydra
from omegaconf import DictConfig
import logging
import numpy as np
from PIL import Image
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.models import resnet18, resnet34
from torchvision import transforms

from models import CIFAR_Network, ImageNet_Network
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
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


def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort

jigsaw_dict = {
    (0, 1, 2, 3): 0,
    (0, 1, 3, 2): 1,
    (0, 2, 1, 3): 2,
    (0, 2, 3, 1): 3,
    (0, 3, 1, 2): 4,
    (0, 3, 2, 1): 5,
    (1, 0, 2, 3): 6,
    (1, 0, 3, 2): 7,
    (1, 2, 0, 3): 8,
    (1, 2, 3, 0): 9,
    (1, 3, 0, 2): 10,
    (1, 3, 2, 0): 11,
    (2, 0, 1, 3): 12,
    (2, 0, 3, 1): 13,
    (2, 1, 0, 3): 14,
    (2, 1, 3, 0): 15,
    (2, 3, 0, 1): 16,
    (2, 3, 1, 0): 17,
    (3, 0, 1, 2): 18,
    (3, 0, 2, 1): 19,
    (3, 1, 0, 2): 20,
    (3, 1, 2, 0): 21,
    (3, 2, 0, 1): 22,
    (3, 2, 1, 0): 23,
}

# Classes for Equivariant Tasks
class HorizontalFlip(torch.nn.Module):
    def __init__(self, base_transform=None, post_transform=transforms.ToTensor(), args=None, anneal=False):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        self.args = args
    
    def forward(self, x):
        r = np.random.randint(2)
        if self.base_transform is not None:
            x = self.base_transform(x)
        t = transforms.RandomHorizontalFlip(p=r)
        x = t(x)
        x = self.post_transform(x)
        return x, r

class VerticalFlip(torch.nn.Module):
    def __init__(self, base_transform=None, post_transform=transforms.ToTensor(), args=None, anneal=False):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        self.args = args
    
    def forward(self, x):
        r = np.random.randint(2)
        if self.base_transform is not None:
            x = self.base_transform(x)
        t = transforms.RandomVerticalFlip(p=r)
        x = t(x)
        x = self.post_transform(x)
        return x, r

class Grayscale(torch.nn.Module):
    def __init__(self, base_transform=None, post_transform=transforms.ToTensor(), args=None, anneal=False):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        self.args = args
    
    def forward(self, x):
        r = np.random.randint(2)
        if self.base_transform is not None:
            x = self.base_transform(x)
        t = transforms.RandomGrayscale(p=r)
        x = t(x)
        x = self.post_transform(x)
        return x, r

class Rotation(torch.nn.Module):
    def __init__(self, base_transform=None, post_transform=transforms.ToTensor(), args=None, anneal=False):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        self.args = args
    
    def forward(self, x):
        r = np.random.randint(4)
        if self.base_transform is not None:
            x = self.base_transform(x)   
        x = transforms.functional.rotate(x, r*90)
        x = self.post_transform(x)
        return x, r


class ColorInversion(torch.nn.Module):
    def __init__(self, base_transform=None, post_transform=transforms.ToTensor(), args=None, anneal=False):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        self.args = args
    
    def forward(self, x):
        r = np.random.randint(2)
        if self.base_transform is not None:
            x = self.base_transform(x)   
        t = transforms.RandomInvert(p=r)
        x = t(x)
        x = self.post_transform(x)
        return x, r

class GaussianBlur(torch.nn.Module):
    def __init__(self, base_transform=None, post_transform=transforms.ToTensor(), args=None, anneal=False):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        self.args = args

    def forward(self, x):
        r = np.random.randint(4)
        if self.base_transform is not None:
            x = self.base_transform(x)
        if r==1:
            t = transforms.GaussianBlur(kernel_size=(5, 5))   
            x = t(x)
        elif r==2:
            t = transforms.GaussianBlur(kernel_size=(9, 9))
            x = t(x)
        elif r==3:
            t = transforms.GaussianBlur(kernel_size=(15, 15))
            x = t(x)
        x = self.post_transform(x)
        return x, r

class Jigsaw(torch.nn.Module):
    def __init__(self, base_transform=None, post_transform=transforms.ToTensor(), args=None, anneal=False):
        super().__init__()
        self.base_transform = base_transform
        self.post_transform = post_transform
        self.args = args
    
    def forward(self, x):
        if self.base_transform is not None:
            x = self.base_transform(x)   
        crop_height, crop_width = 16, 16
        crops = [
            x[:, i*crop_height:(i+1)*crop_height, j*crop_width:(j+1)*crop_width]
            for i in range(2) for j in range(2)
        ]
        shuffle_order = torch.randperm(4)
        shuffled_crops = [crops[i] for i in shuffle_order]
        x = torch.cat([torch.cat([shuffled_crops[0], shuffled_crops[1]], dim=2),
                        torch.cat([shuffled_crops[2], shuffled_crops[3]], dim=2)], dim=1)
        r = jigsaw_dict[tuple(shuffle_order.tolist())]
        return x, r
    

@hydra.main(config_path='./', config_name='simclr_config.yml')
def train(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir
    
    # setting logs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir,'equivariant_tasks'), exist_ok=True)
    log_file_path = os.path.join(script_dir, 'equivariant_tasks', '{}-{}'.format(args.method, args.dataset))
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # prepare the dataset
    if  args.dataset == 'cifar10':
        data_func = CIFAR10
        n_classes = 10
    elif args.dataset == 'cifar100':
        data_func = CIFAR100
        n_classes = 100
    
    # prepare base and post transformations
    base_transform, post_transform = transforms.RandomCrop(size=32), transforms.ToTensor()
    
    # apply different augmentations to images
    if args.method == 'horizontal_flips':
        projection_dim = 2
        print('Using horizontal flips')
        train_transform = HorizontalFlip(base_transform=base_transform, post_transform=post_transform, args=args)
    elif args.method == 'vertical_flips':
        projection_dim = 2
        print('Using vertical flips')
        train_transform = VerticalFlip(base_transform=base_transform, post_transform=post_transform, args=args)
    elif args.method == 'color_inversions':
        projection_dim = 2
        print('Using color inversions')
        train_transform = ColorInversion(base_transform=base_transform, post_transform=post_transform, args=args)
    elif args.method == 'four_fold_rotation':
        projection_dim = 4
        print('Using four-fold rotation')
        train_transform = Rotation(base_transform=base_transform, post_transform=post_transform, args=args)
    elif args.method == 'jigsaws':
        projection_dim = 24
        print('Using jigsaws')
        base_transform = transforms.Compose([transforms.RandomResizedCrop(size=32), transforms.ToTensor()])
        train_transform = Jigsaw(base_transform=base_transform, post_transform=post_transform, args=args)
    elif args.method == 'four_fold_blurs':
        projection_dim = 4
        print('Using four-fold blurs')
        train_transform = GaussianBlur(base_transform=base_transform, post_transform=post_transform, args=args)
    elif args.method == 'grayscale':
        projection_dim = 2
        print('Using grayscale')
        train_transform = Grayscale(base_transform=base_transform, post_transform=post_transform, args=args)
    else:
        projection_dim = 128
        train_transform = transforms.Compose([transforms.RandomResizedCrop(size=32), transforms.ToTensor()])
    
    test_transform = transforms.ToTensor() 
    
    train_set = data_func(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = data_func(root=data_dir, train=False, transform=test_transform, download=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # prepare the model
    assert args.backbone in ['resnet18', 'resnet50']
    base_encoder = eval(args.backbone)
    model = CIFAR_Network(base_encoder, 
                   projection_dim=projection_dim, 
                   projector_type=args.projector_type, 
                   n_classes=n_classes,
                   separate_proj=False,
                   args=args,
                   ).cuda()

    logger.info('Augmentation: {}'.format(args.method))
    logger.info('Backbone: {}, Dataset: {}'.format(args.backbone, args.dataset))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, projection_dim))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    # cosine annealing lr
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step,
            args.epochs * len(train_loader),
            args.learning_rate,
            1e-3))
    
    optimizer_cls = torch.optim.SGD(
        model.linear.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)

    # cosine annealing lr
    scheduler_cls = LambdaLR(
        optimizer_cls,
        lr_lambda=lambda step: get_lr(
            step,
            args.epochs * len(train_loader),
            args.learning_rate,  
            1e-3))

    # pretraining starts
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter("loss")
        acc_meter = AverageMeter("acc")
        train_bar = tqdm(train_loader)
        for x, y in train_bar:
            if args.method != 'none':
                optimizer.zero_grad()
                optimizer_cls.zero_grad()
                x, r = x
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); r = r.cuda(non_blocking=True)
                feature, rep, logits = model(x)
                equi_loss = F.cross_entropy(rep, r)
                cls_loss = F.cross_entropy(logits, y)
                loss = equi_loss + cls_loss
                acc = (rep.argmax(dim=1)==r).float().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
                feature, rep, logits = model(x)
                optimizer.zero_grad()
                optimizer_cls.zero_grad()
                equi_loss = F.cross_entropy(logits, y)
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1)==y).float().mean()
                loss.backward()
                optimizer_cls.step()
                scheduler_cls.step()
            
            loss_meter.update(equi_loss.item(), x.size(0))
            acc_meter.update(acc.item(), x.size(0))
            train_bar.set_description("Train epoch {}, loss: {:.4f}, acc: {:.4f}".format(epoch, loss_meter.avg, acc_meter.avg))
        
        logger.info("Train epoch {}, loss: {:.4f}, acc: {:.4f}".format(epoch, loss_meter.avg, acc_meter.avg))

        # save checkpoint every log_interval epochs
        if epoch >= args.log_interval and epoch % args.log_interval == 0:
            model.eval()
            test_loss_meter = AverageMeter('Test loss')
            test_acc_meter = AverageMeter('Test acc')
            loader_bar = tqdm(test_loader)
            for x, y in loader_bar:
                x, y = x.cuda(), y.cuda()
                logits = model(x, eval_only=True)[-1]
                loss = F.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                test_loss_meter.update(loss.item(), x.size(0))
                test_acc_meter.update(acc.item(), x.size(0))
            
            loader_bar.set_description("Test epoch {}, loss: {:.4f}, acc: {:.4f}".format(epoch, test_loss_meter.avg, test_acc_meter.avg))
            logger.info("Test epoch {}, loss: {:.4f}, acc: {:.4f}".format(epoch, test_loss_meter.avg, test_acc_meter.avg))
            if epoch % args.save_interval == 0:
                os.makedirs(os.path.join(script_dir, args.checkpoint))
                save_path = os.path.join(script_dir, args.checkpoint, '{}-{}-epoch{}.pt'.format(args.method, args.backbone, epoch))
                torch.save(model.state_dict(), save_path)
    logger.info("Training complete for augmentation: {}".format(args.method))

if __name__ == '__main__':
    train()
    
