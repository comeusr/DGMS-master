import torch
import torch.nn.functional as F

import wandb

import config as cfg
from QuantAttention import CustomizGPT2SdpaAttention

from composer.core import Algorithm

def sigmoid_derivative(x):
    return F.sigmoid(x)*(1-F.sigmoid(x))

class GPT2_PRUNER():

    def __init__(self, model, init_sparsity, final_sparsity, alpha_f):
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.cur_sparsity = 0
        self.f_alpha = 0.1
        self.alpha_f = alpha_f
        self.model = model

    def caculate_mask_thresh(self, model, sparsity):
        # Calculuate the pruning threshold for a given sparsity
        # Smaller Pruning Parameters have higher chance to be pruned
        # Ex: Finial_spasity 0.8, then prune the smallest 80% parameters
        is_dict = {}
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2SdpaAttention):
                is_dict[name+'.c_attn'] = m.c_attn.sub_distribution.pruning_parameter.detach()
                is_dict[name+'.c_proj'] = m.c_proj.sub_distribution.pruning_parameter.detach()

        
        all_is = torch.cat([is_dict[name].view(-1) for name in is_dict])

        mask_thresh = torch.kthvalue(all_is, int(sparsity*all_is.shape[0]))[0].item()
        return mask_thresh, is_dict

    def apply_pruning_grad(self, model):
        
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, CustomizGPT2SdpaAttention):
                    # print("Applying sparsisty Gradients")
                    sp=0.01
                    attnLayer = m.c_attn.sub_distribution
                    projLayer = m.c_proj.sub_distribution

                    attnP = attnLayer.pruning_parameter/cfg.PRUNE_SCALE
                    attnLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(attnP)/(sp))*sigmoid_derivative(attnP))

                    projP = projLayer.pruning_parameter/cfg.PRUNE_SCALE
                    projLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(projP)/(sp))*sigmoid_derivative(projP))


                    attnMu = attnLayer.mu
                    attnMu.grad.add_(attnMu, alpha=1/(attnLayer.init_sigma ** 2))

                    projMu = projLayer.mu
                    projMu.grad.add_(projMu, alpha=1/(projLayer.init_sigma ** 2))


        return      
    
    def generate_mask(self, model, mask_thresh, is_dict):
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2SdpaAttention):
                m.c_attn.sub_distribution.mask = (is_dict[name+'.c_attn'] < mask_thresh)
                m.c_proj.sub_distribution.mask = (is_dict[name+'.c_proj'] < mask_thresh)

        return 
    
    def sparsity_scheduler(self, train_step):
        if cfg.PRUNE_START_STEP < train_step <= cfg.PRUNE_END_STEP:
            _frac = 1-(train_step-cfg.PRUNE_START_STEP)/(cfg.PRUNE_END_STEP-cfg.PRUNE_START_STEP)
            sparsity = self.final_sparsity + (self.init_sparsity-self.final_sparsity) * (_frac ** 3)
            self.cur_sparsity = sparsity
        else:
            sparsity = self.final_sparsity
            self.cur_sparsity = sparsity
        return sparsity
    
    def apply_mu_sigma_grad(self, model):
         with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, CustomizGPT2SdpaAttention):
                    attnLayer = m.c_attn.sub_distribution

                    attnMu = attnLayer.mu
                    attnMu.grad.add_(attnMu/(attnLayer.init_sigma ** 2))

                    attnSigma = attnLayer.sigma
                    attnSigma.grad.add_(attnSigma/(attnLayer.init_sigma ** 2)- 1/attnSigma)

                    projLayer = m.c_proj.sub_distribution

                    projMu = projLayer.mu
                    projMu.grad.add_(projMu/(projLayer.init_sigma ** 2))

                    projSigma = projLayer.sigma
                    projSigma.grad.add_(projSigma/(projLayer.init_sigma ** 2)- 1/projSigma)


    
    def pruning_grad_true(self, model):
        # Set the pruning parameter grad equal to True
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2SdpaAttention):
                m.c_attn.sub_distribution.pruning_parameter.requires_grad=True
                m.c_proj.sub_distribution.pruning_parameter.requires_grad=True

    
    def pruning_grad_false(self, model):
        # Set the pruning parameter grad equal to False
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2SdpaAttention):
                m.c_attn.sub_distribution.pruning_parameter.requires_grad=False
                m.c_proj.sub_distribution.pruning_parameter.requires_grad=False

    def prune_with_mask(self, model):
        for name, m in model.named_modules():
            if isinstance(m, CustomizGPT2SdpaAttention):
                attnMask = m.c_attn.sub_distribution.mask
                m.c_attn.sub_distribution.pruning_parameter.detach().masked_fill_(attnMask, -0.1)
                projMask = m.c_proj.sub_distribution.mask
                m.c_proj.sub_distribution.pruning_parameter.detach().masked_fill_(projMask, -0.1)



    def monitor_scheduler_step(self, optimizer):
        for i in range(len(optimizer.param_groups)):
            lr = optimizer.param_groups[i]['lr']
            wandb.log({'parameter_{}_lr'.format(i):lr}, commit=False)

        return
    

    def prune(self, step):

        if cfg.PRUNE and (step <= cfg.PRUNE_START_STEP or step > cfg.PRUNE_END_STEP):
            self.pruning_grad_false(self.model)
        elif cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP:
            # Set Pruning parameter trainable
            self.pruning_grad_true(self.model)
            # Calculate the curr sparsity
            self.sparsity_scheduler(step)
            # Generate mask threshold and help dictionary 
            # is_dict =  {'layer_name': pruning_parameter}
            mask_threshold, is_dict = self.caculate_mask_thresh(self.model, self.cur_sparsity)
            # Generate mask for pruning 
            # mask = {'layer_name': bool matrix}
            self.generate_mask(self.model, mask_threshold, is_dict)
            #Prune with mask
            self.prune_with_mask(self.model)
    
    def apply_non_prune_gradient(self, step):
        try:
            if cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP:
                self.apply_pruning_grad(self.model)
            elif cfg.PRUNE and (step <= cfg.PRUNE_START_STEP or step > cfg.PRUNE_END_STEP):
                self.apply_mu_sigma_grad(self.model)
        except:
            prune_in_progress = cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP
            if prune_in_progress:
                print("Pruning in progress.")
            else:
                print("Not in Pruning.")
            
            print("Attention Require Grad or Not")
            for name, m in self.model.named_modules():
                if isinstance(m, CustomizGPT2SdpaAttention):
                    print(m.c_attn.sub_distribution.pruning_parameter.requires_grad)

    
    def log_sparsity(self):
        wandb.log({'Sparsity': self.cur_sparsity}, commit=False)
    
    