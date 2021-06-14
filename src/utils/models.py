from typing import Callable, List, Dict, Tuple, Sequence, NewType

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import DistilBertModel, BertModel


## Modules ############################################################################################

### BERT Module

class BertPooler(nn.Module):
    def __init__(self, config, final_size, pooling, pool_activation):
        super().__init__()
        self.pooling = pooling
        self.dense = nn.Linear(config.hidden_size, final_size)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.pool_activation = pool_activation

    def forward(self, hidden_states):
        if self.pooling == "CLS":
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            pooled = hidden_states[:, 0]
        elif self.pooling == "MEAN":
            pooled = hidden_states.mean(axis=1)
        # Then we run it through a linear layer and optional activation     
        pooled_output = self.dense(pooled)
        if self.pool_activation == 'tanh':
            pooled_output = self.tanh(pooled_output)
        elif self.pool_activation == 'gelu':
            pooled_output = self.gelu(pooled_output)
            
        return pooled_output
    
class PreTrainedBertPooler(nn.Module):
    def __init__(self, config, pooling):
        super().__init__()
        self.pooling = pooling

    def forward(self, hidden_states):
        if self.pooling == "CLS":
            # We "pool" the model by simply taking the hidden state corresponding
            # to the first token.
            pooled = hidden_states[:, 0]
        elif self.pooling == "MEAN":
            pooled = hidden_states.mean(axis=1)     
        return pooled
    
## Models ############################################################################################

class TripletSingleBERTModel(nn.Module):
    def __init__(self, final_size, pooling, bert_path, model_type='distilbert', pool_activation=None): 
        super().__init__()
        if model_type == 'distilbert':
            self.bert = DistilBertModel.from_pretrained(bert_path, return_dict=True)
        else:
            self.bert = BertModel.from_pretrained(bert_path, return_dict=True)
        self.pooler = BertPooler(self.bert.config, final_size, pooling, pool_activation)

    def forward(self, a, p, n, a_mask, p_mask, n_mask):
        output_a = self.bert(a, attention_mask=a_mask).last_hidden_state
        output_p = self.bert(p, attention_mask=p_mask).last_hidden_state
        output_n = self.bert(n, attention_mask=n_mask).last_hidden_state
        
        output_a = self.pooler(output_a)
        output_p = self.pooler(output_p)
        output_n = self.pooler(output_n)

        return output_a, output_p, output_n 
    
    def return_emb(self, a, a_mask):
        output_a = self.bert(a, attention_mask=a_mask).last_hidden_state
        output_a = self.pooler(output_a)
        return output_a
    
    def downstream_embedding(self, inputs, input_mask):
        output = self.bert(inputs, attention_mask=input_mask).last_hidden_state
        output = self.pooler(output)
        return output

class FrozenSingleBERTModel(nn.Module):
    def __init__(self, final_size, pooling, bert_path, model_type='distilbert', pool_activation=None): 
        super().__init__()
        if model_type == 'distilbert':
            self.bert = DistilBertModel.from_pretrained(bert_path, return_dict=True)
        else:
            self.bert = BertModel.from_pretrained(bert_path, return_dict=True)
            
        for param in self.bert.parameters():
            param.requires_grad = False    
        self.pooler = BertPooler(self.bert.config, final_size, pooling, pool_activation)

    def forward(self, a, p, n, a_mask, p_mask, n_mask):
        output_a = self.bert(a, attention_mask=a_mask).last_hidden_state
        output_p = self.bert(p, attention_mask=p_mask).last_hidden_state
        output_n = self.bert(n, attention_mask=n_mask).last_hidden_state
        
        output_a = self.pooler(output_a)
        output_p = self.pooler(output_p)
        output_n = self.pooler(output_n)

        return output_a, output_p, output_n 
    
    def return_emb(self, a, a_mask):
        output_a = self.bert(a, attention_mask=a_mask).last_hidden_state
        output_a = self.pooler(output_a)
        return output_a
    
    def downstream_embedding(self, inputs, input_mask):
        output = self.bert(inputs, attention_mask=input_mask).last_hidden_state
        output = self.pooler(output)
        return output
    
class PreTrainedBERTModel(nn.Module):
    def __init__(self, final_size = None, pooling = 'CLS', 
                 bert_path=None, model_type='distilbert', pool_activation=None): 
        super().__init__()
        if model_type == 'distilbert':
            self.bert = DistilBertModel.from_pretrained(bert_path, return_dict=True)
        else:
            self.bert = BertModel.from_pretrained(bert_path, return_dict=True)
        self.pooler = PreTrainedBertPooler(self.bert.config, pooling)
        
    def forward(self, a, p, n, a_mask, p_mask, n_mask):
        output_a = self.bert(a, attention_mask=a_mask).last_hidden_state
        output_p = self.bert(p, attention_mask=p_mask).last_hidden_state
        output_n = self.bert(n, attention_mask=n_mask).last_hidden_state
        
        output_a = self.pooler(output_a)
        output_p = self.pooler(output_p)
        output_n = self.pooler(output_n)

        return output_a, output_p, output_n 
    
    def return_emb(self, a, a_mask):
        output_a = self.bert(a, attention_mask=a_mask).last_hidden_state
        output_a = self.pooler(output_a)
        return output_a
    
    def downstream_embedding(self, inputs, input_mask):
        output = self.bert(inputs, attention_mask=input_mask).last_hidden_state
        return output
    
    
class TripletDoubleBERTModel(nn.Module):
    def __init__(self, final_size, pooling, bert_path, model_type='distilbert', pool_activation=None): 
        super().__init__()
        if model_type == 'distilbert':
            self.bert_l = DistilBertModel.from_pretrained(bert_path, return_dict=True)
            self.bert_r = DistilBertModel.from_pretrained(bert_path, return_dict=True)
        else:
            self.bert_l = BertModel.from_pretrained(bert_path, return_dict=True)
            self.bert_r = BertModel.from_pretrained(bert_path, return_dict=True)
        self.pooler_l = BertPooler(self.bert_l.config, final_size, pooling, pool_activation)
        self.pooler_r = BertPooler(self.bert_r.config, final_size, pooling, pool_activation)

    def forward(self, a, p, n, a_mask, p_mask, n_mask):
        output_a = self.bert_l(a, attention_mask=a_mask).last_hidden_state
        output_p = self.bert_r(p, attention_mask=p_mask).last_hidden_state
        output_n = self.bert_r(n, attention_mask=n_mask).last_hidden_state
        
        output_a = self.pooler_l(output_a)
        output_p = self.pooler_r(output_p)
        output_n = self.pooler_r(output_n)

        return output_a, output_p, output_n 
    
    def return_emb_l(self, a, a_mask):
        output_a = self.bert_l(a, attention_mask=a_mask).last_hidden_state
        output_a = self.pooler_l(output_a)
        return output_a
    
    def return_emb_r(self, a, a_mask):
        output_a = self.bert_r(a, attention_mask=a_mask).last_hidden_state
        output_a = self.pooler_r(output_a)
        return output_a
    
    def downstream_embedding(self, inputs, input_mask):
        output = self.bert(inputs, attention_mask=input_mask).last_hidden_state
        output = self.pooler(output)
        return output