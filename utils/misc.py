import errno
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import config as cfg
from utils.cluster import kmeans, kmeans_predict
from utils.lr_scheduler import get_scheduler
from sklearn.mixture import GaussianMixture
from modeling.SQS import SQSConv
from torch_kmeans import KMeans
from composer.models import ComposerModel


__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'cluster_weights', 'get_optimizer', 'resume_ckpt']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

@torch.no_grad()
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()

@torch.no_grad()
def check_cuda_memory():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print("Total CUDA Memory {:.3f} GBs, Used Memory {:.3f} GBs".format(reserved, allocated))

@torch.no_grad()
def cluster_weights(weights, n_clusters, iter_limit=100):
    """ Initialization of GMM with k-means algorithm, note this procedure may bring
    different initialization results, and the results may be slightly different.
    Args:
        weights:[weight_size]
        n_clusters: 1 + 2^i = K Gaussian Mixture Component number
    Returns:
        [n_clusters-1] initial region saliency obtained by k-means algorthm
    """
    flat_weight = weights.view(-1, 1).cuda()
    _tol = 1e-11
    if cfg.IS_NORMAL is True or cfg.DEBUG:
        print("skip k-means")
        tmp = torch.rand(n_clusters-1).cuda()
        return tmp, tmp , 0.5, tmp, 0.01
    _cluster_idx, region_saliency = kmeans(X=flat_weight, num_clusters=n_clusters, tol=_tol, \
                        distance='euclidean', iter_limit=iter_limit, device=torch.device('cuda'), tqdm_flag=False)
    pi_initialization = torch.tensor([torch.true_divide(_cluster_idx.eq(i).sum(), _cluster_idx.numel()) \
                            for i in range(n_clusters)], device='cuda')
    zero_center_idx = torch.argmin(torch.abs(region_saliency))
    region_saliency_tmp = region_saliency.clone()
    region_saliency_zero = region_saliency[zero_center_idx]
    region_saliency_tmp[zero_center_idx] = 0.0
    pi_zero = pi_initialization[zero_center_idx]

    sigma_tmp = torch.zeros(n_clusters,1).cuda()
    for i in range(flat_weight.size(0)):
        _idx = _cluster_idx[i]
        sigma_tmp[_idx] += (flat_weight[i,0]-region_saliency_tmp[_idx])**2
    sigma_initialization = torch.tensor([torch.true_divide(sigma_tmp[i], _cluster_idx.eq(i).sum()-1) \
                                    for i in range(n_clusters)], device='cuda').sqrt()
    sigma_zero = sigma_initialization[zero_center_idx]
    sigma_initialization = sigma_initialization[torch.arange(region_saliency.size(0)).cuda() != zero_center_idx]    

    pi_initialization = pi_initialization[torch.arange(region_saliency.size(0)).cuda() != zero_center_idx]
    region_saliency = region_saliency[torch.arange(region_saliency.size(0)).cuda() != zero_center_idx] # remove zero component center
    return region_saliency, pi_initialization, pi_zero, sigma_initialization, sigma_zero

@torch.no_grad()
def cluster_weights_sparsity(weights, n_clusters, iter_limit=100):
    """ Initialization of GMM with k-means algorithm, note this procedure may bring
    different initialization results, and the results may be slightly different.
    Args:
        weights:[weight_size]
        n_clusters: 1 + 2^i = K Gaussian Mixture Component number
    Returns:
        [n_clusters-1] initial region saliency obtained by k-means algorthm
    """
    flat_weight = weights.view(-1, 1).to(DEVICE)
    _tol = 1e-11
    if cfg.IS_NORMAL is True or cfg.DEBUG:
        print("skip k-means")
        tmp = torch.rand(n_clusters).to(DEVICE)
        return tmp, tmp, tmp
    _cluster_idx, region_saliency = kmeans(X=flat_weight, num_clusters=n_clusters, tol=_tol, \
                        distance='euclidean', iter_limit=iter_limit, device=torch.device('cuda'), tqdm_flag=False)
    # Kmean_model = Kmeans(n_clusters=n_clusters, max_iter=300, tol=_tol)
    # result = Kmean_model(flat_weight)
    # _cluster_idx, region_saliency = result.label, result.cluster_centers
    
    pi_initialization = torch.tensor([torch.true_divide(_cluster_idx.eq(i).sum(), _cluster_idx.numel()) \
                            for i in range(n_clusters)], device='cuda')
    # zero_center_idx = torch.argmin(torch.abs(region_saliency))
    region_saliency_tmp = region_saliency.clone()
    # region_saliency_zero = region_saliency[zero_center_idx]
    # region_saliency_tmp[zero_center_idx] = 0.0
    # pi_zero = pi_initialization[zero_center_idx]

    sigma_tmp = torch.zeros(n_clusters,1).to(DEVICE)
    # # for i in range(n_clusters):
    #     indices = _cluster_idx.index(i)
    #     temp = flat_weight[indices,0]
    #     sigma_tmp[i] =
    for i in range(flat_weight.size(0)):
        _idx = _cluster_idx[i]
        sigma_tmp[_idx] += (flat_weight[i,0]-region_saliency_tmp[_idx])**2
    sigma_initialization = torch.tensor([torch.true_divide(sigma_tmp[i], _cluster_idx.eq(i).sum()-1) \
                                    for i in range(n_clusters)], device='cuda').sqrt()
    # sigma_zero = sigma_initialization[zero_center_idx]
    # sigma_initialization = sigma_initialization[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx]

    # pi_initialization = pi_initialization[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx]
    # region_saliency = region_saliency[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx] # remove zero component center
    # print('Sigma shape {}'.format(sigma_initialization.shape))
    # print('Sigma_zero shape {}'.format(sigma_initialization.shape))
    # print('Sigma_initialization shape {}'.format(sigma_initialization.shape))
    # print('Pi_initialization shape {}'.format(pi_initialization.shape))
    # print('Region_saliency shape return {}'.format(region_saliency.shape))

    return region_saliency, pi_initialization, sigma_initialization

@torch.no_grad()
def cluster_weights_em(weights, n_clusters):
    """ Initialization of GMM with EM algorithm, note this procedure may bring
    different initialization results, and the results may be slightly different.
    Args:
        weights:[weight_size]
        n_clusters: 1 + 2^i = K Gaussian Mixture Component number
    Returns:
        [n_clusters-1] initial region saliency obtained by k-means algorthm
    """
    flat_weight = weights.view(-1, 1).contiguous().detach().numpy()
    _tol = 1e-5
    if cfg.IS_NORMAL is True:
        print("skip k-means")
        tmp = torch.rand(n_clusters-1).to(DEVICE)
        return tmp, tmp , 0.5, tmp, 0.01
    # construct GMM using EM algorithm
    gm = GaussianMixture(n_components=n_clusters, random_state=0, tol=_tol).fit(flat_weight)
    region_saliency = torch.from_numpy(gm.means_).view(-1).to(DEVICE)
    pi_initialization = torch.from_numpy(gm.weights_).to(DEVICE)
    sigma_initialization = torch.from_numpy(gm.covariances_).view(-1).sqrt().to(DEVICE)
    
    zero_center_idx = torch.argmin(torch.abs(region_saliency))
    pi_zero = pi_initialization[zero_center_idx]
    sigma_zero = sigma_initialization[zero_center_idx]
    sigma_initialization = sigma_initialization[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx] 
    pi_initialization = pi_initialization[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx]
    region_saliency = region_saliency[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx] # remove zero component center
    return region_saliency, pi_initialization, pi_zero, sigma_initialization, sigma_zero


@torch.no_grad()
def cluster_weight_quantile(weights, n_clusters):
    q = torch.linspace(0,1, n_clusters+2)[1:-1]
    print("Printing Quantiles")
    # print(q)
    flat_weight = weights.view(-1, 1).contiguous().detach().numpy()
    region_saliency = torch.quantile(flat_weight, q)
    # print(region_saliency)
    _cluster_idx = kmeans_predict(flat_weight, region_saliency)

    pi_initialization = torch.tensor([torch.true_divide(_cluster_idx.eq(i).sum(), _cluster_idx.numel()) \
                            for i in range(n_clusters)], device='cuda')
    zero_center_idx = torch.argmin(torch.abs(region_saliency))
    region_saliency_tmp = region_saliency.clone()
    region_saliency_zero = region_saliency[zero_center_idx]
    region_saliency_tmp[zero_center_idx] = 0.0
    pi_zero = pi_initialization[zero_center_idx]

    sigma_tmp = torch.zeros(n_clusters,1).to(DEVICE)

    for i in range(flat_weight.size(0)):
        _idx = _cluster_idx[i]
        sigma_tmp[_idx] += (flat_weight[i,0]-region_saliency_tmp[_idx])**2
    sigma_initialization = torch.tensor([torch.true_divide(sigma_tmp[i], _cluster_idx.eq(i).sum()-1) \
                                    for i in range(n_clusters)], device='cuda').sqrt()
    sigma_zero = sigma_initialization[zero_center_idx]
    sigma_initialization = sigma_initialization[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx]    

    pi_initialization = pi_initialization[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx]
    region_saliency = region_saliency[torch.arange(region_saliency.size(0)).to(DEVICE) != zero_center_idx] # remove zero component center
    return region_saliency, pi_initialization, pi_zero, sigma_initialization, sigma_zero
    

def get_optimizer(model, args):
    train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}]
    optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=args.nesterov)
    return optimizer

@torch.no_grad()
def freeze_param(model):
    for name, m in model.named_modules():
        if isinstance(m, SQSConv):
            m.weight.requires_grad=False

def resume_ckpt(args, model, train_loader, optimizer, lr_scheduler):
    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
    if args.only_inference:
        checkpoint = torch.load(args.resume)
        if args.cuda:
            model.module.load_state_dict(checkpoint)
            model = model.to(DEVICE)
        else:
            model.load_state_dict(checkpoint['state_dict'])
            model.init_mask_params()
            optimizer = get_optimizer(model, args)
            lr_scheduler = get_scheduler(args, optimizer, \
                    args.lr, len(train_loader))
        best_pred = 0.0
        print("=> loaded checkpoint '{}'".format(args.resume))
    else:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if args.cuda:
            model.module.load_state_dict(checkpoint['state_dict'])
            model.module.init_mask_params()
            optimizer = get_optimizer(model.module, args)
            lr_scheduler = get_scheduler(args, optimizer, \
                    args.lr, len(train_loader))
            model = model.to(DEVICE)
        else:
            model.load_state_dict(checkpoint['state_dict'])
            model.init_mask_params()
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.rt:
            best_pred = 0.0
        else:
            best_pred = checkpoint['best_pred']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        print("Checkpoint Top-1 Acc: ", best_pred)
    return model, optimizer, lr_scheduler, best_pred

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L363-L384
    """
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
