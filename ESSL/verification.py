import hydra
from omegaconf import DictConfig
import logging
import numpy as np
from PIL import Image
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
from utils import *

from models import CIFAR_Network, ImageNet_Network
from tqdm import tqdm


logger = logging.getLogger(__name__)


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

class Rotation(torch.nn.Module):
    def __init__(self, base_transform=transforms.ToTensor(), post_transform=transforms.ToTensor(), args=None, anneal=False):
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

@hydra.main(config_path='./', config_name='simclr_config.yml')
def train(args: DictConfig) -> None:
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    # setting logs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir,'verification'), exist_ok=True)
    log_file_path = os.path.join(script_dir, 'verification', '{}-{}'.format(args.method, args.dataset))
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    data_dir = hydra.utils.to_absolute_path(args.data_dir)  # get absolute path of data dir
    if  args.dataset == 'cifar10':
        data_func = CIFAR10
        n_classes = 10
    elif args.dataset == 'cifar100':
        data_func = CIFAR100
        n_classes = 100

    args.projection_dim = 4
    base_transform, post_transform = transforms.RandomCrop(32), transforms.ToTensor()
    train_transform = Rotation(base_transform=base_transform, post_transform=post_transform, args=args)
    test_transform = transforms.ToTensor() 
    
    train_set = data_func(root=data_dir, train=True, transform=train_transform, download=True)
    test_set = data_func(root=data_dir, train=False, transform=test_transform, download=False)

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # prepare the model
    assert args.backbone in ['resnet18', 'resnet50']
    base_encoder = eval(args.backbone)
    model = CIFAR_Network(base_encoder, 
                   projection_dim=args.projection_dim, 
                   projector_type=args.projector_type, 
                   n_classes=n_classes,
                   separate_proj=args.method.startswith('Mixed'),
                   args=args,
                   ).cuda()
    
    logger.info('Backbone: {}, Dataset: {}'.format(args.backbone, args.dataset))
    logger.info('method: {}, lambda: {}'.format(args.method, args.lmbd))
    logger.info('feature dim: {}, projection dim: {}'.format(model.feature_dim, args.projection_dim))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    
    optimizer_classifier = torch.optim.SGD(
        model.linear.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True)
    
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step,
            args.epochs * len(train_loader),
            args.learning_rate,
            1e-3))
    
    scheduler_classifier = LambdaLR(
        optimizer_classifier,
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
            optimizer.zero_grad()
            optimizer_classifier.zero_grad()
            if args.method == 'normal':
                x, r = x
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); r = r.cuda(non_blocking=True)
                feature, rep, logits = model(x)
                rot_loss = F.cross_entropy(rep, r)
                loss = rot_loss
                acc = (rep.argmax(dim=1)==r).float().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
            elif args.method == 'add':
                x, r = x
                l = args.lmbd
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); r = r.cuda(non_blocking=True)
                feature, rep, logits = model(x)
                rot_loss = F.cross_entropy(rep, r)
                loss = F.cross_entropy(rep, r) + l * F.cross_entropy(logits, y)
                acc = (rep.argmax(dim=1)==r).float().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
            elif args.method == 'eliminate':
                x, r = x
                l = args.lmbd
                x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True); r = r.cuda(non_blocking=True)
                feature, rep, logits = model(x)
                rot_loss = F.cross_entropy(rep, r)
                loss = F.cross_entropy(rep, r) - l * F.cross_entropy(logits, y)
                acc = (rep.argmax(dim=1)==r).float().mean()
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()
                cls_loss = F.cross_entropy(logits, y)
                cls_loss.backward()
                optimizer_classifier.step()
                scheduler_classifier.step()
            else:
                raise ValueError('Unsupported method')

            loss_meter.update(rot_loss.item(), x.size(0))
            acc_meter.update(acc.item(), x.size(0))
            train_bar.set_description("Train epoch {}, Rot loss: {:.4f} Train ACC: {:.4f}".format(epoch, loss_meter.avg, acc_meter.avg))
        logger.info("Train epoch {}, Rot loss: {:.4f} Train ACC: {:.4f}".format(epoch, loss_meter.avg, acc_meter.avg))

        # save checkpoint very log_interval epochs
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
                loader_bar.set_description("Test epoch {}, Cls loss: {:.4f} Test ACC: {:.4f}".format(epoch, test_loss_meter.avg, test_acc_meter.avg))
            logger.info("Test epoch {}, Cls loss: {:.4f} Test ACC: {:.4f}".format(epoch, test_loss_meter.avg, test_acc_meter.avg))
            if epoch % args.save_interval == 0:
                os.makedirs(os.path.join(script_dir, args.checkpoint))
                save_path = os.path.join(script_dir, args.checkpoint, '{}-{}-epoch{}.pt'.format(args.method, args.backbone, epoch))
                torch.save(model.state_dict(), save_path)

    logger.info("Training complete for method: {}".format(args.method))


if __name__ == '__main__':
    train()

