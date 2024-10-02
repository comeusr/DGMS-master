from composer.core import Algorithm, Event
from composer.models import ComposerModel
from composer import State
import wandb
import config as cfg
import torch
import math
import torch.nn.functional as F

from QuantAttention import CustomizeBertSelfAttention, CustomizeBertSelfOutput

def sigmoid_derivative(x):
    return F.sigmoid(x)*(1-F.sigmoid(x))

class BERT_PRUNER(Algorithm):
    
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
            if isinstance(m, CustomizeBertSelfAttention):
                is_dict[name+'.query'] = m.query.sub_distribution.pruning_parameter.detach()
                is_dict[name+'.key'] = m.key.sub_distribution.pruning_parameter.detach()
                is_dict[name+'.value'] = m.value.sub_distribution.pruning_parameter.detach()
            elif isinstance(m, CustomizeBertSelfOutput):
                is_dict[name+'.dense'] = m.dense.sub_distribution.pruning_parameter.detach()

        
        all_is = torch.cat([is_dict[name].view(-1) for name in is_dict])
       
        mask_thresh = torch.kthvalue(all_is, int(sparsity*all_is.shape[0]))[0].item()
        return mask_thresh, is_dict

    def apply_pruning_grad(self, model):
        
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, CustomizeBertSelfAttention):
                    # print("Applying sparsisty Gradients")
                    sp=0.01
                    queryLayer = m.query.sub_distribution
                    keyLayer = m.key.sub_distribution
                    valueLayer = m.value.sub_distribution

                    queryp = queryLayer.pruning_parameter/cfg.PRUNE_SCALE
                    queryLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(queryp)/(sp))*sigmoid_derivative(queryp))

                    keyp = keyLayer.pruning_parameter/cfg.PRUNE_SCALE
                    keyLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(keyp)/(sp))*sigmoid_derivative(keyp))

                    valuep = valueLayer.pruning_parameter/cfg.PRUNE_SCALE
                    valueLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(valuep)/(sp))*sigmoid_derivative(valuep))
                    

                    queryMu = queryLayer.mu
                    queryMu.grad.add_(queryMu, alpha=1/(queryLayer.init_sigma ** 2))

                    keyMu = keyLayer.mu
                    keyMu.grad.add_(keyMu, alpha=1/(keyLayer.init_sigma ** 2))

                    valueMu = valueLayer.mu
                    valueMu.grad.add_(valueMu, alpha=1/(valueLayer.init_sigma ** 2))
                elif isinstance(m, CustomizeBertSelfOutput):
                    sp=0.01
                    outputLayer = m.dense.sub_distribution
                    outputp = outputLayer.pruning_parameter/cfg.PRUNE_SCALE
                    outputLayer.pruning_parameter.grad.add_(torch.log(F.sigmoid(outputp)/(sp))*sigmoid_derivative(outputp))
        return      
    
    def generate_mask(self, model, mask_thresh, is_dict):
        for name, m in model.named_modules():
            if isinstance(m, CustomizeBertSelfAttention):
                m.query.sub_distribution.mask = (is_dict[name+'.query'] < mask_thresh)
                m.key.sub_distribution.mask = (is_dict[name+'.key'] < mask_thresh)
                m.value.sub_distribution.mask = (is_dict[name+'.value'] < mask_thresh)
            elif isinstance(m, CustomizeBertSelfOutput):
                m.dense.sub_distribution.mask = (is_dict[name+'.dense'] < mask_thresh)

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
                if isinstance(m, CustomizeBertSelfAttention):
                    # print("Applying sparsisty Gradients")
                    queryLayer = m.query.sub_distribution

                    queryMu = queryLayer.mu
                    queryMu.grad.add_(queryMu/(queryLayer.init_sigma ** 2))

                    querySigma = queryLayer.sigma
                    querySigma.grad.add_(querySigma/(queryLayer.init_sigma ** 2)- 1/querySigma)

                    keyLayer = m.key.sub_distribution

                    keyMu = keyLayer.mu
                    keyMu.grad.add_(keyMu/(keyLayer.init_sigma ** 2))

                    keySigma = keyLayer.sigma
                    keySigma.grad.add_(keySigma/(keyLayer.init_sigma ** 2)- 1/keySigma)

                    valueLayer = m.value.sub_distribution

                    valueMu = valueLayer.mu
                    valueMu.grad.add_(valueMu/(valueLayer.init_sigma ** 2))

                    valueSigma = valueLayer.sigma
                    valueSigma.grad.add_(valueSigma/(valueLayer.init_sigma ** 2)- 1/valueSigma)
                elif isinstance(m, CustomizeBertSelfOutput):
                    outputLayer = m.dense.sub_distribution

                    queryMu = queryLayer.mu
                    queryMu.grad.add_(queryMu/(queryLayer.init_sigma ** 2))



    
    def pruning_grad_true(self, model):
        # Set the pruning parameter grad equal to True
        for name, m in model.named_modules():
            if isinstance(m, CustomizeBertSelfAttention):
                m.query.sub_distribution.pruning_parameter.requires_grad=True
                m.value.sub_distribution.pruning_parameter.requires_grad=True
                m.key.sub_distribution.pruning_parameter.requires_grad=True
            elif isinstance(m, CustomizeBertSelfOutput):
                m.dense.sub_distribution.pruning_parameter.requires_grad=True

    
    def pruning_grad_false(self, model):
        # Set the pruning parameter grad equal to False
        for name, m in model.named_modules():
            if isinstance(m, CustomizeBertSelfAttention):
                m.query.sub_distribution.pruning_parameter.requires_grad=False
                m.value.sub_distribution.pruning_parameter.requires_grad=False
                m.key.sub_distribution.pruning_parameter.requires_grad=False
            elif isinstance(m, CustomizeBertSelfOutput):
                m.dense.sub_distribution.pruning_parameter.requires_grad=False

    def prune_with_mask(self, model):
        for name, m in model.named_modules():
            if isinstance(m, CustomizeBertSelfAttention):
                queryMask = m.query.sub_distribution.mask
                m.query.sub_distribution.pruning_parameter.detach().masked_fill_(queryMask, -0.1)
                keyMask = m.key.sub_distribution.mask
                m.key.sub_distribution.pruning_parameter.detach().masked_fill_(keyMask, -0.1)
                valueMask = m.value.sub_distribution.mask
                m.value.sub_distribution.pruning_parameter.detach().masked_fill_(valueMask, -0.1)
            elif isinstance(m, CustomizeBertSelfOutput):
                _Mask = m.dense.sub_distribution.mask
                m.dense.sub_distribution.pruning_parameter.detach().masked_fill_(_Mask, -0.1)


    def monitor_scheduler_step(self, optimizer):
        # print(optimzier)
        for i in range(len(optimizer.param_groups)):
            lr = optimizer.param_groups[i]['lr']
            wandb.log({'parameter_{}_lr'.format(i):lr}, commit=False)

        return
    
    
    def match(self, event, state):
        return event in [Event.BEFORE_TRAIN_BATCH, Event.AFTER_BACKWARD, Event.BATCH_START]
    

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
        if cfg.PRUNE and cfg.PRUNE_START_STEP < step <= cfg.PRUNE_END_STEP:
            self.apply_pruning_grad(self.model)
        elif cfg.PRUNE and (step <= cfg.PRUNE_START_STEP or step > cfg.PRUNE_END_STEP):
            self.apply_mu_sigma_grad(self.model)
    
    def log_sparsity(self):
        wandb.log({'Sparsity': self.cur_sparsity}, commit=False)
    
    def apply(self, event, state, logger):
        train_step = state.timestamp.batch.value
        # if cfg.PRUNE and cfg.PRUNE_START_STEP > 0:
        # TO DO
        # Apply the KL-divergence gradients
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
    