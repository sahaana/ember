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

from src.replearn.encoder import eval_model
from src.utils.datasets import DemoDataset, DemoHardNegativeDataset, EmberEvalDataset
from src.utils.models import TripletSingleBERTModel
from src.utils.embedding_utils import tokenize_batch, tokenize_single_data_batch, save_torch_model, to_cuda
from src.utils.file_utils import param_header, make_dir
from src.utils.knn_utils import FaissKNeighbors, compute_top_k_pd, knn_metrics


def get_cuda_levels(batch_tokenizer):
    levels = {'tokenize_columns': 2,
              'tokenize_batch': 1, 
              'tokenize_string_categorical': 0,
              'tokenize_single_data_batch': 0}
    return levels[batch_tokenizer.__name__]


def perform_knn(left, right, test_supervision, latest_model_path, conf):
    model = TripletSingleBERTModel(conf.final_size, conf.pool_type, conf.bert_path)
    model.load_state_dict(torch.load(latest_model_path))
    
    #if conf.data in ['imdb_wiki', 'SQuAD_sent', 'MSMARCO', 'deepmatcher', 'small_imdb_fuzzy', 'hard_imdb_fuzzy',
    #                 'main_fuzzy', 'hard_fuzzy', 'easy_fuzzy', 'dm_blocked']:
    #    left = pd.read_pickle(conf.eval_datapath_l)
    #    right = pd.read_pickle(conf.eval_datapath_r)
    #    test_supervision = pd.read_pickle(conf.test_supervision)
    #if conf.data == 'MSMARCO':
    #left = left.set_index(conf.ID_left)
    #right = right.set_index(conf.ID_right)
    
    tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer)   
    left_eval_data = DataLoader(EmberEvalDataset(left, conf.new_col_name, indexed=True), 
                                batch_size=conf.batch_size,
                                shuffle = False
                               )
    right_eval_data = DataLoader(EmberEvalDataset(right, conf.new_col_name, indexed=True), 
                                 batch_size=conf.batch_size,
                                 shuffle = False
                                )
    #if conf.arch == 'double-triplet':
    #    left_index, left_embeddings = eval_model(model, tokenizer, left_eval_data, conf.tokenizer_max_length, mode='LEFT')
    #    right_index, right_embeddings = eval_model(model, tokenizer, right_eval_data, conf.tokenizer_max_length, mode='RIGHT')
    #else:
    left_index, left_embeddings = eval_model(model, tokenizer, left_eval_data, conf.tokenizer_max_length)
    right_index, right_embeddings = eval_model(model, tokenizer, right_eval_data, conf.tokenizer_max_length)
    
    knn = FaissKNeighbors(k=conf.right_size)
    knn.fit(right_embeddings)
    neib = knn.kneighbors(left_embeddings)
    
    knn_routine_params = (neib[0], neib[1], test_supervision, left_index, right_index, conf.ID_left, conf.ID_right)    
    knn_results = compute_top_k_pd(knn_metrics, knn_routine_params, k_max=conf.right_size, thresh=None)   
    
    save_dir = param_header(conf.batch_size, conf.final_size, conf.lr, conf.pool_type, conf.epochs, conf.train_size)
    save_dir = f'{conf.data_path}/results/{conf.model_name}/{save_dir}/knn/'
    make_dir(save_dir)
    timestamp = datetime.now().strftime('%H-%M-%d-%m-%y')
    save_path = f"{save_dir}/{timestamp}_knn_results.pkl"
    knn_results.to_pickle(save_path)
    print(f"Saved Results: {save_path}")
    
    embedding_out = {"left_index": left_index,
                     "left_embeddings": left_embeddings,
                     "right_index": right_index,
                     "right_embeddings": right_embeddings}
    
    with open(f'{save_dir}/{timestamp}_embeddings.pkl', 'wb') as handle:
        pickle.dump(embedding_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return embedding_out


