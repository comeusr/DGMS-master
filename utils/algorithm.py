from composer.core import Algorithm, Event
from composer.models import ComposerModel
from composer import State
from modeling.SQS import SQSConv
import config as cfg
import torch
import math
import torch.nn.functional as F

def sigmoid_derivative(x):
    return F.sigmoid(x)*(1-F.sigmoid(x))

class GMM_Pruning(Algorithm):
    
    def __init__(self, init_sparsity, final_sparsity, alpha_f):
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.cur_sparsity = 0
        self.f_alpha = 0.1
        self.alpha_f = alpha_f
        # self.pruning_scaling = 
        

    def caculate_mask_thresh(self, model: ComposerModel, sparsity):
        # Calculuate the pruning threshold for a given sparsity
        # Smaller Pruning Parameters have higher chance to be pruned
        # Ex: Finial_spasity 0.8, then prune the smallest 80% parameters
        is_dict = {}
        for name, m in model.named_modules():
            if isinstance(m, SQSConv):
                is_dict[name] = m.sub_distribution.pruning_parameter.detach()
                # print("is_dict_{} {}".format(name, is_dict[name]))
        
        all_is = torch.cat([is_dict[name].view(-1) for name in is_dict])
       
        mask_thresh = torch.kthvalue(all_is, int(sparsity*all_is.shape[0]))[0].item()
        return mask_thresh, is_dict

    def apply_pruning_grad(self, model: ComposerModel):
        
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, SQSConv):
                    # print("Applying sparsisty Gradients")
                    sp=0.01
                    layer = m.sub_distribution
                    p = layer.pruning_parameter/cfg.PRUNE_SCALE
                    layer.pruning_parameter.grad.add_(torch.log(F.sigmoid(p)/(sp))*sigmoid_derivative(p))

                    mu = layer.mu
                    mu.grad.add_(mu, alpha=1/(layer.init_sigma ** 2))
        return      
    
    def generate_mask(self, model:ComposerModel, mask_thresh, is_dict):
        for name, m in model.named_modules():
            if isinstance(m, SQSConv):
                m.sub_distribution.mask = (is_dict[name] < mask_thresh)

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
                if isinstance(m, SQSConv):
                    layer = m.sub_distribution

                    mu = layer.mu
                    mu.grad.add_(mu/(layer.init_sigma ** 2))

                    sigma = layer.sigma
                    sigma.grad.add_(sigma/(layer.init_sigma ** 2)- 1/sigma)


    
    def pruning_grad_true(self, model):
        # Set the pruning parameter grad equal to True
        for name, m in model.named_modules():
            if isinstance(m, SQSConv):
                m.sub_distribution.pruning_parameter.requires_grad=True
    
    def pruning_grad_false(self, model):
        # Set the pruning parameter grad equal to False
        for name, m in model.named_modules():
            if isinstance(m, SQSConv):
                m.sub_distribution.pruning_parameter.requires_grad=False

    def prune_with_mask(self, model):
        for name, m in model.named_modules():
            if isinstance(m, SQSConv):
                mask = m.sub_distribution.mask
                m.sub_distribution.pruning_parameter.detach().masked_fill_(mask, -0.1)


    def monitor_scheduler_step(self, state:State, logger):
        optimzier = state.optimizers[0]
        for i in range(len(optimzier.param_groups)):
            lr = optimzier.param_groups[i]['lr']
            logger.log_metrics({'parameter_{}_lr'.format(i):lr})

        return
    
    
    def match(self, event, state):
        return event in [Event.BEFORE_TRAIN_BATCH, Event.AFTER_BACKWARD, Event.BATCH_START]
    
    def apply(self, event, state, logger):
        train_step = state.timestamp.batch.value

        if event == Event.BEFORE_TRAIN_BATCH:
            # Prune the parameter according to the pruning parameters
            if cfg.PRUNE and (train_step <= cfg.PRUNE_START_STEP or train_step > cfg.PRUNE_END_STEP):
                self.pruning_grad_false(state.model)
            elif cfg.PRUNE and cfg.PRUNE_START_STEP < train_step <= cfg.PRUNE_END_STEP:
                # Set Pruning parameter trainable
                self.pruning_grad_true(state.model)
                # Calculate the curr sparsity
                self.sparsity_scheduler(train_step)
                # Generate mask threshold and help dictionary 
                # is_dict =  {'layer_name': pruning_parameter}
                mask_threshold, is_dict = self.caculate_mask_thresh(state.model, self.cur_sparsity)
                # Generate mask for pruning 
                # mask = {'layer_name': bool matrix}
                self.generate_mask(state.model, mask_threshold, is_dict)
                #Prune with mask
                self.prune_with_mask(state.model)
            
                            
        elif event == Event.AFTER_BACKWARD:
            # Add the gradients of KL divergence to pruning parameters
            # print("Apply Pruning Gradient")
            if cfg.PRUNE and cfg.PRUNE_START_STEP < train_step <= cfg.PRUNE_END_STEP:
                self.apply_pruning_grad(state.model)
            elif cfg.PRUNE and (train_step <= cfg.PRUNE_START_STEP or train_step > cfg.PRUNE_END_STEP):
                self.apply_mu_sigma_grad(state.model)
        elif event == event.BATCH_START:
            logger.log_metrics({'sparsity': self.cur_sparsity})

        return
    