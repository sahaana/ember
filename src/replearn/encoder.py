import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from types import SimpleNamespace   
from collections import defaultdict

import numpy as np
import pandas as pd

import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DistilBertModel

from src.utils.datasets import DemoDataset, DemoHardNegativeDataset
from src.utils.models import TripletSingleBERTModel
from src.utils.embedding_utils import tokenize_batch, tokenize_single_data_batch, save_torch_model, to_cuda
from src.utils.file_utils import param_header

#from src.utils.file_utils import load_config


def get_cuda_levels(batch_tokenizer):
    levels = {'tokenize_columns': 2,
              'tokenize_batch': 1, 
              'tokenize_string_categorical': 0,
              'tokenize_single_data_batch': 0}
    return levels[batch_tokenizer.__name__]

def train_emb_model(model, 
                    tokenizer,
                    batch_tokenizer, 
                    train_data, 
                    loss_func, 
                    optimizer, 
                    epochs, 
                    save_dir,
                    tokenizer_max_length = 512,
                    train_model = True) -> str:
    
    levels = get_cuda_levels(batch_tokenizer)
    model.cuda()
    for epoch in range(epochs):
        if train_model:
            wandb.log({"Epoch": epoch}) 
            model.train()

            for i, d in enumerate(train_data):
                batch = batch_tokenizer(d, tokenizer, tokenizer_max_length)
                inputs, masks = to_cuda(batch, levels=levels)
                a, p, n = inputs
                a_mask, p_mask, n_mask = masks
                optimizer.zero_grad()
                oa, op, on = model(a, p, n, a_mask, p_mask, n_mask)
                loss = loss_func(oa, op, on)
                loss.backward()
                optimizer.step()

                if (i % 100) == 0 : 
                    wandb.log({"train batch loss": loss.item()})
                if (i % 20000 == 0):
                    save_torch_model(save_dir, model)
        last_saved = save_torch_model(save_dir, model)
    return last_saved
                
                
def eval_model(model, 
               tokenizer,
               data,
               tokenizer_max_length = 512, 
               mode = None): #mode can be right or left too, but make sure to only use it with the double tower models
    embeddings = []
    model.eval()
    model.cuda()
    
    if mode == 'LEFT':
        emb_return = model.return_emb_l
    elif mode == 'RIGHT':
        emb_return = model.return_emb_r
    else:
        emb_return = model.return_emb
    for i, d in enumerate(data):
        batch = tokenize_single_data_batch(d, tokenizer, tokenizer_max_length)
        inputs, masks = to_cuda(batch, levels=0)
        out = emb_return(inputs, masks)
        embeddings.append(out.cpu().detach().numpy())
    embeddings = np.vstack(embeddings)
    if data.dataset.indexed:
        return np.array(data.dataset.index), embeddings
    return embeddings


def train_embedding(left, right, train_supervision, train_negatives, conf: SimpleNamespace):
    if 'pos_frac' not in conf.__dict__:
        conf.pos_frac = 1
        
    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer)
    #bert_model = DistilBertModel.from_pretrained(conf.bert_path, return_dict=True)
         
    #if 'negatives' in conf.model_name:
    train_data =  DataLoader(DemoHardNegativeDataset(left, right, conf.train_size, conf.new_col_name, train_supervision, params = {'negatives': train_negatives, 'pos_frac': conf.pos_frac}, left_id = conf.ID_left, right_id = conf.ID_right), 
                        batch_size=conf.batch_size,
                        shuffle = True
                        )  
    #else: 
    #    train_data = DataLoader(DemoDataset(left, right, train_size, conf.new_col_name, train_supervision, params = {'pos_frac': conf.pos_frac}), 
    #                            batch_size=conf.batch_size,
    #                            shuffle = True
    #                            )
    
    if conf.loss == 'triplet':
        loss = nn.TripletMarginLoss(margin=conf.tl_margin, p=conf.tl_p)
        model = TripletSingleBERTModel(final_size = conf.final_size, pooling = conf.pool_type, bert_path = conf.bert_path)
        optimizer = optim.AdamW(model.parameters(), lr=conf.lr)#optim.SGD(model.parameters(), lr=lr)
    
    save_dir = param_header(conf.batch_size, conf.final_size, conf.lr, conf.pool_type, conf.epochs, conf.train_size)
    save_dir = f'{conf.data_path}/models/emb/{conf.model_name}/{save_dir}/'
    wandb.init(project=conf.model_name)
    
    print("Training Begins")
    last_saved = train_emb_model(model, 
                                 tokenizer, 
                                 tokenize_batch, 
                                 train_data, 
                                 loss, 
                                 optimizer, 
                                 conf.epochs, 
                                 save_dir, 
                                 conf.tokenizer_max_length,
                                 train_model=True)
    
    return last_saved
