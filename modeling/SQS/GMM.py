""" DGMS GM Sub-distribution implementation.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config as cfg

from utils.misc import cluster_weights, get_device, cluster_weights_sparsity

DEVICE = get_device()


class GaussianMixtureModel(nn.Module):
    """Concrete GMM for sub-distribution approximation.
    """
    def __init__(self, num_components, init_weights, temperature=0.01, init_method="k-means", init_sigma=3):
        super(GaussianMixtureModel, self).__init__()
        self.num_components = num_components
        self.temperature = temperature
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.params_initialization(init_weights, init_method)
        self.prune = cfg.PRUNE
        self.mask = (init_weights.abs()< 0.0).to(DEVICE)
        # print('GMM weight dim {}'.format(init_weights.shape))
        # print('Init Mask dim {}'.format(self.mask.shape))
        if cfg.PRUNE:
            self.init_sigma = init_sigma

    def params_initialization(self, init_weights, method='k-means'):
        if not cfg.PRUNE:
            """ Initialization of GMM parameters using k-means algorithm. """
            self.mu_zero = torch.tensor([0.0], device=self.device).float()
            self.pi_k, self.mu, self.sigma = \
                    torch.ones(self.num_components-1, device=self.device), \
                    torch.ones(self.num_components-1, device=self.device), \
                    torch.ones(self.num_components-1, device=self.device)
            if method == 'k-means':
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            elif method == "quantile":
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
            elif method == 'empirical':
                initial_region_saliency, pi_init, pi_zero_init, sigma_init, _sigma_zero = cluster_weights(init_weights, self.num_components)
                sigma_init, _sigma_zero = torch.ones_like(sigma_init).mul(0.01).to(DEVICE), torch.ones_like(torch.tensor([_sigma_zero])).mul(0.01).to(DEVICE)
            self.mu = nn.Parameter(data=torch.mul(self.mu.to(DEVICE), initial_region_saliency.flatten().to(DEVICE)))
            self.pi_k = nn.Parameter(data=torch.mul(self.pi_k.to(DEVICE), pi_init)).to(DEVICE).float()
            self.pi_zero = nn.Parameter(data=torch.tensor([pi_zero_init], device=self.device)).to(DEVICE).float()
            self.sigma_zero = nn.Parameter(data=torch.tensor([_sigma_zero], device=self.device)).float()
            self.sigma = nn.Parameter(data=torch.mul(self.sigma, sigma_init)).to(DEVICE).float()
            self.temperature = nn.Parameter(data=torch.tensor([self.temperature], device=self.device), requires_grad=False)
        else:
            """ Intialization of GMM + Pruning parameters using k-means"""
            self.mu_zero = torch.tensor([0.0], device=self.device).float()
            self.pi_k, self.mu, self.sigma = \
                    torch.ones(self.num_components, device=self.device), \
                    torch.ones(self.num_components, device=self.device), \
                    torch.ones(self.num_components, device=self.device)
            if method == 'k-means':
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
            elif method == "quantile":
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
            elif method == 'empirical':
                initial_region_saliency, pi_init, sigma_init = cluster_weights_sparsity(init_weights, self.num_components)
                sigma_init = torch.ones_like(sigma_init).mul(0.01).to(DEVICE)
                # sigma_init, _sigma_zero = torch.ones_like(sigma_init).mul(0.01).to(DEVICE), torch.ones_like(torch.tensor([_sigma_zero])).mul(0.01).to(DEVICE)
            
            # initial_region_saliency = pi_init = sigma_init = torch.ones_like(self.mu, device='cuda')
            self.mu = nn.Parameter(data=torch.mul(self.mu.to(DEVICE), initial_region_saliency.flatten().to(DEVICE)))
            self.pi_k = nn.Parameter(data=torch.mul(self.pi_k.to(DEVICE), pi_init)).to(DEVICE).float()
            # self.pi_zero = nn.Parameter(data=torch.tensor([pi_zero_init], device=self.device)).to(DEVICE).float()
            # self.sigma_zero = nn.Parameter(data=torch.tensor([_sigma_zero], device=self.device)).float()
            self.sigma = nn.Parameter(data=torch.mul(self.sigma, sigma_init)).to(DEVICE).float()
            self.temperature = nn.Parameter(data=torch.tensor([self.temperature], device=self.device), requires_grad=False)
            self.pruning_parameter = nn.Parameter(data=5*cfg.PRUNE_SCALE*torch.ones_like(init_weights, device=self.device))

    def gaussian_mixing_regularization(self):
        # pi_tmp = torch.cat([self.pi_zero, self.pi_k], dim=-1).abs()
        if not cfg.PRUNE:
            pi_tmp = torch.cat([self.pi_zero, self.pi_k], dim=-1).abs()
            res = torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).cuda()
            # print('Pi shape{}'.format(res.shape))
            return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).cuda()
        else:
            pi_tmp = self.pi_k.abs()
            return torch.div(pi_tmp, pi_tmp.sum(dim=-1).unsqueeze(-1)).to(DEVICE)

    def Normal_pdf(self, x, _pi, mu, sigma):
        """ Standard Normal Distribution PDF. """
        return torch.mul(torch.reciprocal(torch.sqrt(torch.mul( \
               torch.tensor([2 * math.pi], device=self.device), sigma**2))), \
               torch.exp(-torch.div((x - mu)**2, 2 * sigma**2))).mul(_pi)

    def GMM_region_responsibility(self, weights):
        if not cfg.PRUNE:
            """" Region responsibility of GMM. """
            pi_normalized = self.gaussian_mixing_regularization().cuda()
            responsibility = torch.zeros([self.num_components, weights.size(0)], device=self.device)
            responsibility[0] = self.Normal_pdf(weights.cuda(), pi_normalized[0], 0.0, self.sigma_zero.cuda())
            for k in range(self.num_components-1):
                responsibility[k+1] = self.Normal_pdf(weights, pi_normalized[k+1], self.mu[k].cuda(), self.sigma[k].cuda())
            responsibility = torch.div(responsibility, responsibility.sum(dim=0) + cfg.EPS)
            return F.softmax(responsibility / self.temperature, dim=0)
        else:
            """" Region responsibility of GMM. """
            pi_normalized = self.gaussian_mixing_regularization().to(DEVICE)
            responsibility = torch.zeros([self.num_components, weights.size(0)], device=self.device)
            # responsibility[0] = self.Normal_pdf(weights.to(DEVICE), pi_normalized[0], 0.0, self.sigma_zero.to(DEVICE))
            for k in range(self.num_components):
                responsibility[k] = self.Normal_pdf(weights, pi_normalized[k], self.mu[k].to(DEVICE), self.sigma[k].to(DEVICE))
            responsibility = torch.div(responsibility, responsibility.sum(dim=0) + cfg.EPS)
            return F.softmax(responsibility / self.temperature, dim=0)


    def forward(self, weights, train=True):
        if not cfg.PRUNE:
            if train:
                # soft mask generalized pruning during training
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("Printing the region_belong shape {}".format(self.region_belonging.shape))
                Sweight = torch.mul(self.region_belonging[0], 0.) \
                        + torch.mul(self.region_belonging[1:], self.mu.unsqueeze(1)).sum(dim=0)
                return Sweight.view(weights.size())
            else:
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("Printing the region_belong shape {}".format(self.region_belonging.shape))
                max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
                mask_w = torch.zeros_like(self.region_belonging).scatter_(dim=0, index=max_index, value=1.)
                Pweight = torch.mul(mask_w[1:], self.mu.unsqueeze(1)).sum(dim=0)
                return Pweight.view(weights.size())
        else:
            if train:
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("torch.mul(self.region_belonging[0], 0.) {}".format(torch.mul(self.region_belonging[0], 0.)))
                # print("torch.mul(self.region_belonging, self.mu.unsqueeze(1)).sum(dim=0) * F.sigmoid(self.pruning_parameter.flatten()) {}".format(torch.mul(self.region_belonging, self.mu.unsqueeze(1)).sum(dim=0) * F.sigmoid(self.pruning_parameter.flatten())))
                # print("Printing the region_belong shape {}".format(self.region_belonging.shape))               
                Sweight = torch.mul(self.region_belonging[0], 0.) \
                        + torch.mul(self.region_belonging, self.mu.unsqueeze(1)).sum(dim=0) * F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE)
                # print('Pruning Scaler {}'.format(F.sigmoid(self.pruning_parameter.flatten()/cfg.PRUNE_SCALE)))
                return Sweight.view(weights.size())
            else:
                self.region_belonging = self.GMM_region_responsibility(weights.flatten())
                # print("Printing the region_belong shape {}".format(self.region_belonging.shape))               
                if cfg.SAMPLE:
                    # max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
                    max_index = self.region_belonging.transpose(0, 1).multinomial(num_samples=1).transpose(0, 1)
                else:
                    max_index = torch.argmax(self.region_belonging, dim=0).unsqueeze(0)
                # print("Print the max_index shape {}".format(max_index.shape))
                mask_w = torch.zeros_like(self.region_belonging).scatter_(dim=0, index=max_index, value=1.)
                Pweight = torch.mul(mask_w, self.mu.unsqueeze(1)).sum(dim=0)
                # print('Pweight before mask {}'.format(Pweight))
                Pweight = Pweight.view(weights.size())
                # print("Mask Shape {}".format(self.mask.shape))
                # print("Masked size {}".format(self.mask.sum()))
                Pweight.detach().masked_fill_(self.mask, 0.0)
                return Pweight

def gmm_approximation(num_components, init_weights, temperature=0.5, init_method='k-means', sigma=3) -> GaussianMixtureModel:
    return GaussianMixtureModel(num_components, init_weights, temperature, init_method, sigma)