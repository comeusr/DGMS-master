from composer.core import Algorithm, Event
from composer.models import ComposerModel
from modeling.DGMS import DGMSConv
import config as cfg
import torch
import torch.nn.functional as F

def sigmoid_derivative(x):
    return F.sigmoid(x)*(1-F.sigmoid(x))

class GMM_Pruning(Algorithm):
    
    def __init__(self, init_sparsity, final_sparsity):
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        

    def caculate_mask_thresh(self, model: ComposerModel, sparsity):
        # Calculuate the pruning threshold for a given sparsity
        # Smaller Pruning Parameters have higher chance to be pruned
        # Ex: Finial_spasity 0.8, then prune the smallest 80% parameters
        is_dict = {}
        for name, m in model.named_modules():
            if isinstance(m, DGMSConv):
                is_dict[name] = m.sub_distribution.pruning_parameter.detach()
        
        all_is = torch.cat([is_dict[name].view(-1) for name in is_dict])
        mask_thresh = torch.kthvalue(all_is, int(sparsity*all_is.shape[0]))[0].item()
        return mask_thresh, is_dict

    def apply_pruning_grad(self, model: ComposerModel):
        
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, DGMSConv):
                    p = m.sub_distribution.pruning_parameter
                    m.sub_distribution.pruning_parameter.grad.add_(torch.log(F.sigmoid(p.detach())/(1-self.cur_sparsity))*sigmoid_derivative(p.detach()))
        return
    
    def generate_mask(self, model:ComposerModel, mask_thresh, is_dict):
        for name, m in model.named_modules():
            if isinstance(m, DGMSConv):
                m.sub_distribution.mask = (is_dict[name] < mask_thresh)
        return 
    
    def sparsity_scheduler(self, train_step):
        _frac = 1-(train_step)/(cfg.PRUNE_END_STEP)
        sparsity = self.final_sparsity + (self.init_sparsity-self.final_sparsity) * (_frac ** 3)
        self.cur_sparsity = sparsity
        return sparsity
        

    def prune_with_mask(self, model):
        for name, m in model.named_modules():
            if isinstance(m, DGMSConv):
                mask = m.sub_distribution.mask
                m.sub_distribution.pruning_parameter.detach().masked_fill_(mask, 0.0)
    
    def match(self, event, state):
        return event in [Event.BEFORE_TRAIN_BATCH, Event.AFTER_BACKWARD]
    
    def apply(self, event, state, logger):
        train_step = state.timestamp.batch.value
        # TO DO
        # Apply the KL-divergence gradients
        if event == Event.BEFORE_TRAIN_BATCH:
            # Prune the parameter according to the pruning parameters

            # First calculate the curr sparsity
            self.sparsity_scheduler(train_step)
            # Generate mask threshold and help dictionary 
            # is_dict =  {'layer_name': pruning_parameter}
            mask_threshold, is_dict = self.caculate_mask_thresh(state.model, self.cur_sparsity)
            # Generate mask for pruning 
            # mask = {'layer_name': 0, 1 matrix}
            self.generate_mask(state.model, mask_threshold, is_dict)
            #Prune with mask
            self.prune_with_mask(state.model)
            
        elif event == Event.AFTER_BACKWARD:
            # Add the gradients of KL divergence to pruning parameters
            self.apply_pruning_grad(state.model)
            
            return
    