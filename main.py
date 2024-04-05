# Copyright (c) Runpei Dong, ArChip Lab.
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch

import argparse
import time
import os
import sys
import math
import wandb
import torch.nn as nn
import config as cfg
import detectors
import timm
import copy

from tqdm import tqdm
from mypath import Path
from dataloader import make_data_loader
from modeling import DGMSNet
from modeling.DGMS import DGMSConv
from utils.PyTransformer.transformers.torchTransformer import TorchTransformer
from utils.loss import *
from utils.misc import freeze_param
from utils.watch import Sparsity, EpochMonitor

from composer import Trainer
from composer.loggers import WandBLogger, TensorboardLogger
from composer.optim import DecoupledAdamW, LinearWithWarmupScheduler
from composer.callbacks import LRMonitor, OptimizerMonitor, NaNMonitor




def main():
    parser = argparse.ArgumentParser(description="Differentiable Gaussian Mixture Weight Sharing (DGMS)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--network', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'mnasnet', 'proxylessnas',
                                 'resnet20', 'resnet32', 'resnet56', 'vggsmall'],
                        help='network name (default: resnet18)')
    parser.add_argument('-d', '--dataset', type=str, default='imagenet',
                        choices=['cifar10', 'imagenet', 'cars', 'cub200', 'aircraft'],
                        help='dataset name (default: imgenet)')
    parser.add_argument('-j', '--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=32,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=32,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--train-dir', type=str, default=None,
                        help='training set directory (default: None)')
    parser.add_argument('--val-dir', type=str, default='None',
                        help='validation set directory (default: None)')
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='Number of classes (default: 1000)')
    parser.add_argument('--show-info', action='store_true', default=False, 
                        help='set if show model compression info (default: False)')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', 
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=256, metavar='N', 
                        help='input batch size for testing (default: 256)')
    # model params
    parser.add_argument('--K', type=int, default=16, metavar='K',
                        help='number of GMM components (default: 2^4=16)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='TAU',
                        help='gumbel softmax temperature (default: 0.01)')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='whether train noramlly (default: False)')
    parser.add_argument('--empirical', type=bool, default=False,
                        help='whether use empirical initialization for parameter sigma (default: False)')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='whether transform normal convolution into DGMS convolution (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--lr-scheduler', type=str, default='one-cycle',
                        choices=['one-cycle', 'cosine', 'multi-step', 'reduce'],
                        help='lr scheduler mode: (default: one-cycle)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training (default: False)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default="Experiments",
                        help='set the checkpoint name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='set if use a pretrained network')
    # re-train a pre-trained model
    parser.add_argument('--rt', action='store_true', default=False,
                        help='retraining model for quantization')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--only-inference', action='store_true', default=False,
                        help='skip training and only inference')
    parser.add_argument('--wandb_watch', action='store_true', default=False,
                        help='Use Weights & Bias as logger.')
    parser.add_argument('--t_warmup', type=str, default='0.0dur',
                        help="Length of learning rate warm up phase.")
    parser.add_argument('--alpha_f', type=float, default=0.001,
                        help="Final learning rate.")
    parser.add_argument('--duration', type=str, default='200ep',
                        help="Number of Epochs")
    parser.add_argument('--watch_freq', type=int, default=1000,
                        help="Frequency of Wandb watch model")
    parser.add_argument('--run_name', type=str, default=None,
                        help="Run name")
    parser.add_argument('--save_folder', type=str, default=None,
                        help="Save Directory")
    parser.add_argument('--save_interval', type=str, default='0.01dur',
                        help='Frequency of Saving Model')
    parser.add_argument('--load_path', type=str, default=None,
                        help='Directory to resume the training')
    parser.add_argument('--autoresume', action='store_true', default=False,
                        help="Auto Resume the training process.")
    parser.add_argument('--eval_interval', type=str, default='5ep',
                        help="Eval Interval")
    parser.add_argument('--freeze_weight', action='store_true', default=False,
                        help='Freeze Parameters')

    args = parser.parse_args([
        "--train-dir", "/home/wang4538/DGMS-master/CIFAR10/train/", "--val-dir", "/home/wang4538/DGMS-master/CIFAR10/val/", "-d", "cifar10",
        "--num-classes", "10", "--lr", "2e-5",  "--base-size", "32", "--crop-size", "32",
        "--network", "resnet18", "--mask", "--K", "4", "--weight_decay", "5e-4",
        "--empirical", "True", "--tau", "0.01", '--normal', '--freeze_weight',
        "--show-info", "--wandb_watch", "--t_warmup", "0.1dur", "--alpha_f", "0.001", '--eval_interval', '1ep',
        "--duration", "2ep", "--save_folder", "/scratch/gilbreth/wang4538/DGMS/debug/cifar10", "--autoresume", '--run_name', 'debug'
    ])
    # args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # saver = Saver(args)
    train_loader, val_loader, test_loader, nclass = make_data_loader(args)

    # Load Pretrain Data
    model = timm.create_model("resnet18_cifar10", pretrained=True)
    # base_model = copy.deepcopy(model)
    # print("-"*40+"Pretrian Model"+"-"*40)
    # for name, m in model.named_modules():
    #     print(name)

    model = DGMSNet(model, args, args.freeze_bn)


    print("DGMS Conv!")
    _transformer = TorchTransformer()
    _transformer.register(nn.Conv2d, DGMSConv)
    model = _transformer.trans_layers(model)


    print("-" * 40 + "DGMS Model" + "-" * 40)
    if args.freeze_weight:
        for name, m in model.named_modules():
            if isinstance(m, DGMSConv):
                m.weight.requires_grad=False

    # if args.freeze_weight:
    #     freeze_param(model)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.init_mask_params()

    cfg.IS_NORMAL = True if (args.resume is not None) else False
    cfg.IS_NORMAL = args.normal


    optimizer = DecoupledAdamW(
        model.network.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    lr_scheduler = LinearWithWarmupScheduler(
        t_warmup=args.t_warmup,
        alpha_f=args.alpha_f
    )

    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        schedulers=lr_scheduler,
        max_duration=args.duration,
        device_train_microbatch_size='auto',

        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        eval_interval=args.eval_interval,
        device="gpu" if torch.cuda.is_available() else "cpu",

        # callbacks
        callbacks=[EpochMonitor(), LRMonitor(), OptimizerMonitor()],
        # callbacks=[LRMonitor(), OptimizerMonitor()],
        loggers=[WandBLogger()],

        #Save Checkpoint
        # save_overwrite=True,
        save_folder=args.save_folder,
        save_filename="ep{epoch}",
        save_latest_filename="latest",
        autoresume=True,
        # load_path=args.load_path,
        run_name=args.run_name,

        seed=args.seed

    )

    trainer.fit()

    trainer.close()


if __name__ == '__main__':
    main()
