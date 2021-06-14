import sys
import json
import argparse
from types import SimpleNamespace    
from collections import defaultdict 
from typing import Callable, List, Dict, Tuple, Sequence, NewType

import numpy as np
import pandas as pd

from src.utils.preprocessing_utils import compute_BM25, DataCollatorForEnrich
from src.utils.datasets import MLMDataset
from src.utils.file_utils import make_dir


from transformers import AutoTokenizer
from transformers import pipeline, Trainer, TrainingArguments
from transformers import DistilBertConfig, DistilBertForMaskedLM, BertConfig, BertForMaskedLM


def BM25Pretrainer(corpus_df: pd.DataFrame,
                   query_df: pd.DataFrame,
                   conf: SimpleNamespace,
                   save_BM25: bool = True):
    fpath = conf.data_path
    data_col = conf.new_col_name
    make_dir(f'{fpath}/BM25')
    # switching query and corpus idk why it's switched from OG
    bm25 = compute_BM25(query_df, corpus_df, data_col, fpath)
    combined_bm25 = pd.DataFrame(np.hstack([np.array(corpus_df.index)[:,None], bm25[2]]))
    
    combined_bm25 = combined_bm25.set_index(0)
    pd.to_pickle(combined_bm25, f'{fpath}/BM25/BM25_argsort_indices.pkl')
    return combined_bm25


def train_MLM(data_l: pd.DataFrame,
              data_r: pd.DataFrame,
              bm25_argsort: pd.DataFrame, 
              conf: SimpleNamespace):

    bert_tokenizer = AutoTokenizer.from_pretrained(conf.tokenizer)
    data_collator = DataCollatorForEnrich(tokenizer=bert_tokenizer, 
                                          mlm=conf.pretrain_mlm, 
                                          mlm_probability=conf.mlm_probability,
                                          masking=conf.mlm_masking,
                                          num_seps=conf.mlm_num_seps)
    model_out = f'{conf.data_path}/models/MLM/{conf.mlm_model_name}'
    
    # Model 
    if conf.model_type == 'distilbert':
        model_config = DistilBertConfig() 
        if conf.from_scratch:
            model = DistilBertForMaskedLM(config=model_config)
        else:
            model = DistilBertForMaskedLM(config=model_config).from_pretrained(f"{conf.tokenizer}")
    
    elif conf.model_type == 'bert':
        model_config = BertConfig()
        if conf.from_scratch:
            model = BertForMaskedLM(config=model_config)
        else:
            model = BertForMaskedLM(config=model_config).from_pretrained(f"{conf.tokenizer}")
    
    train_data_l = data_l.copy()
    train_data_r = data_r.copy()
    train_bm25 = bm25_argsort.copy()

    train_dataset = MLMDataset(train_data_l, train_data_r, 
                             bert_tokenizer, data_col=conf.new_col_name, 
                             bm25_argsort=train_bm25)  


    training_args = TrainingArguments(output_dir=model_out,
                                      overwrite_output_dir=True,
                                      num_train_epochs=conf.mlm_train_epochs,
                                      per_device_train_batch_size=conf.mlm_batch_size,
                                      save_steps=10_000)

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_dataset)
    
    # Train and save
    trainer.train()
    trainer.save_model(model_out)   
    
    
def compute_bm25_negatives(original_supervision, BM25, conf):
    supervision = original_supervision.set_index(conf.ID_left)
    new_supervision = defaultdict(list)
    for idx, entry in supervision.iterrows():
        new_supervision[conf.ID_left].append(idx)
        new_supervision[conf.ID_right].append([])
        
        for j in range(30):
            new_negative = BM25.loc[idx][len(BM25.columns)-j]
            if new_negative not in entry[conf.ID_right]:
                new_supervision[conf.ID_right][-1].append(new_negative)

    new_supervision = pd.DataFrame(new_supervision)
    new_supervision.to_pickle(f'{conf.data_path}/hard_negatives_supervision_train.pkl')
    return new_supervision



    