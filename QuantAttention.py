from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
import config as cfg
from modeling.SQS.GMM import *

from typing import List, Optional, Tuple, Union

from transformers.models.bert.modeling_bert import load_tf_weights_in_bert, \
    BertSelfAttention, BertSelfOutput
from transformers.models.gpt2.modeling_gpt2 import GPT2SdpaAttention


class CustomizeBertSelfOutput(BertSelfOutput):
    def __init__(self, config):
        super().__init__(config)

        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU
    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.dense.sub_distribution = gmm_approximation(self.k_level, self.dense.weight, self.temperature, init_method, sigma)

    def QuantizedWeights(self):
        # Quantized weight from the given sub distribution.

        if cfg.IS_TRAIN:
            _weight = self.dense.sub_distribution(weights=self.dense.weight, train=True)
        else:
            _weight = self.dense.sub_distribution(weights=self.dense.weight, train=False)
        
        return _weight

    def softforward(self, _weight:torch.Tensor, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, _weight)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        if cfg.IS_NORMAL:
            return super().forward(hidden_states, 
                                   input_tensor)
        else:
            _weight = self.QuantizedWeights()
            return self.softforward(_weight,
                                    hidden_states,
                                    input_tensor)
    
        

class CustomizeBertSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None):
        super().__init__(config, position_embedding_type)

        self.dropout_prob = config.attention_probs_dropout_prob
        self.require_contiguous_qkv = False

        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU
    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.query.sub_distribution = gmm_approximation(self.k_level, self.query.weight, self.temperature, init_method, sigma)
        self.key.sub_distribution = gmm_approximation(self.k_level, self.key.weight, self.temperature, init_method, sigma)
        self.value.sub_distribution = gmm_approximation(self.k_level, self.value.weight, self.temperature, init_method, sigma)

    def getMu(self):
        return (self.query.sub_distribution.mu.detach().data.cpu().numpy(), 
                self.key.sub_distribution.mu.detach().data.cpu().numpy(), 
                self.value.sub_distribution.mu.detach().data.cpu().numpy())
    
    def get_Sweight(self):
        # soft quantized weights during training
        with torch.no_grad():
            return (self.query.sub_distribution(weights=self.query.weight, train=True),
                    self.key.sub_distribution(weights=self.key.weight, train=True),
                    self.value.sub_distribution(weights=self.value.weight, train=True))

    def get_Pweight(self):
        # hard quantized weights during inference
        with torch.no_grad():
            return (self.query.sub_distribution(weights=self.query.weight, train=False),
                    self.key.sub_distribution(weights=self.key.weight, train=False),
                    self.value.sub_distribution(weights=self.value.weight, train=False))
        
    def QuantizedWeights(self):
        # Quantized weight from the given sub distribution.

        if cfg.IS_TRAIN:
            query_weight = self.query.sub_distribution(weights=self.query.weight, train=True)
            key_weight = self.key.sub_distribution(weights=self.key.weight, train=True)
            value_weight = self.value.sub_distribution(weights=self.value.weight, train=True)
        else:
            query_weight = self.query.sub_distribution(weights=self.query.weight, train=False)
            key_weight = self.key.sub_distribution(weights=self.key.weight, train=False)
            value_weight = self.value.sub_distribution(weights=self.value.weight, train=False)

        return query_weight, key_weight, value_weight

    def softforward(self, 
                query_weight: torch.Tensor,
                key_weight: torch.Tensor,
                value_weight: torch.Tensor,
                hidden_states: torch.Tensor, 
                attention_mask: torch.FloatTensor | None = None, 
                head_mask: torch.FloatTensor | None = None, 
                encoder_hidden_states: torch.FloatTensor | None = None, 
                encoder_attention_mask: torch.FloatTensor | None = None, 
                past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None, 
                output_attentions: bool | None = False) -> Tuple[torch.Tensor]:

        mixed_query_layer = F.linear(hidden_states, weight=query_weight)
        # print('Debug Key weight')
        # print(key_weight)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(F.linear(encoder_hidden_states, key_weight))
            value_layer = self.transpose_for_scores(F.linear(encoder_hidden_states, value_weight))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(F.linear(hidden_states, key_weight))
            value_layer = self.transpose_for_scores(F.linear(hidden_states, value_weight))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(F.linear(hidden_states, key_weight))
            value_layer = self.transpose_for_scores(F.linear(hidden_states, value_weight))  
        
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: torch.FloatTensor | None = None, 
                head_mask: torch.FloatTensor | None = None, 
                encoder_hidden_states: torch.FloatTensor | None = None, 
                encoder_attention_mask: torch.FloatTensor | None = None, 
                past_key_value: Tuple[Tuple[torch.FloatTensor]] | None = None, 
                output_attentions: bool | None = False) -> Tuple[torch.Tensor]:
        if cfg.IS_NORMAL:
            return super().forward(hidden_states, 
                                   attention_mask, 
                                   head_mask, 
                                   encoder_hidden_states, 
                                   encoder_attention_mask, 
                                   past_key_value, 
                                   output_attentions)
        else:
            Qweight, Kweight, VWeight = self.QuantizedWeights()
            return self.softforward(Qweight,
                                Kweight,
                                VWeight,
                                hidden_states, 
                                attention_mask, 
                                head_mask, 
                                encoder_hidden_states, 
                                encoder_attention_mask, 
                                past_key_value, 
                                output_attentions)
    

class CustomizGPT2SdpaAttention(GPT2SdpaAttention):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

        self.is_normal = cfg.IS_NORMAL

        self.k_level = cfg.K_LEVEL
        self.temperature = cfg.TAU

    
    def init_mask_params(self, sigma):
        init_method = 'empirical' if cfg.IS_EMP else 'k-means'
        self.c_attn.sub_distribution = gmm_approximation(self.k_level, self.c_attn.weight, self.temperature, init_method, sigma)
        self.c_proj.sub_distribution = gmm_approximation(self.k_level, self.c_proj.weight, self.temperature, init_method, sigma)

        # self.c_attn = nn.Conv1d(in_channels, out_channels)


    def get_Sweight(self):
        # soft quantized weights during training
        with torch.no_grad():
            return (self.c_attn.sub_distribution(weights=self.c_attn.weight, train=True),
                    self.c_proj.sub_distribution(weights=self.c_proj.weight, train=True))
    
    def QuantizedWeights(self):
        if cfg.IS_TRAIN:
            c_attn_weights = self.c_attn.sub_distribution(weights=self.c_attn.weight, train=True)
            c_proj_weights = self.c_proj.sub_distribution(weights=self.c_proj.weight, train=True)
        else:
            c_attn_weights = self.c_attn.sub_distribution(weights=self.c_attn.weight, train=True)
            c_proj_weights = self.c_proj.sub_distribution(weights=self.c_proj.weight, train=True)
        
        return c_attn_weights, c_proj_weights
        

    def softforward(
        self,
        c_attn_weights,
        c_proj_weights,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        bsz, q_len, _ = hidden_states.size()

        # print("Original c_attn weight shape {}".format(self.c_attn.weight.shape))
        # print("Customize c_attn weight shape {}".format(c_attn_weights.shape))

        # Initial attention projections
        is_cross_attention = encoder_hidden_states is not None
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2SdpaAttention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = F.linear(hidden_states, c_attn_weights.transpose(0,1), self.c_attn.bias).split(self.split_size, dim=2)

        # self.c_attn.weight
        # self.c_attn(hidden_states)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Optional kv caching
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        present = None
        if use_cache is True:
            present = (key, value)

        # Avoid torch==2.1.2 specific bug for the memory-efficient backend in SDPA
        if self.require_contiguous_qkv and query.device.type == "cuda" and attention_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if attention_mask is None and q_len > 1 and not is_cross_attention else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape outputs
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.embed_dim)

        # Final projection
        # attn_output = self.c_proj(attn_output)
        attn_output = F.linear(attn_output, c_proj_weights.transpose(0,1), self.c_proj.bias)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present, None
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        
        if self.is_normal:
            return super().forward(
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
        else:
            c_attn_weights, c_proj_weioghts = self.QuantizedWeights()
            return self.softforward(
                c_attn_weights=c_attn_weights,
                c_proj_weights=c_proj_weioghts,
                hidden_states=hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions
            )
    

