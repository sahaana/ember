from dataclasses import dataclass
from typing import List, Union, Dict, Optional, Tuple, Callable, Sequence, NewType

import pandas as pd
import numpy as np

import torch 
from transformers import PreTrainedTokenizerBase, BatchEncoding, AutoTokenizer, BertTokenizer, DistilBertTokenizer

from src.utils.data_utils import sample_excluding


######## MLM ################################################################################


class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 tokenizer: PreTrainedTokenizerBase,
                 n_pairs_multiplier: int = 3,
                 data_col = 'merged',
                 max_len:int = 512, 
                 bm25_argsort: pd.DataFrame = None):
        self.df_l = df_l.copy()[data_col]
        self.df_r = df_r.copy()[data_col]
        self.bm25_argsort = bm25_argsort #in this dataset, the bm25 is prepended with the index of the query
        self.tokenizer = tokenizer
        self.n_pairs = 0 ## this determined based on input; some entries just don't have enough in the top 1000 list given
        self.pairs = self.gen_pairs(n_pairs_multiplier) 
        self.max_len = max_len

    def gen_pairs(self, n_pairs_multiplier):
        pairs = []
        for entry in self.bm25_argsort.iterrows():
            idx_l = entry[0] #in this dataset, the bm25 is prepended with the index of the query
            bm25_entries = list(entry[1])
            for j in range(1, 1 + n_pairs_multiplier):
                idx_r = bm25_entries[-j]
                if idx_r != -1: ## this won't cause issues by going back to the query index becuase the query index is INDEX 
                    pairs.append([idx_l, idx_r])
                    self.n_pairs += 1
        return np.array(pairs)   
        
    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx): ##continue this
        l, r = self.pairs[idx]
        return {'input_ids': self.tokenizer(self.df_l[l] + ' [SEP] ' + self.df_r[r], 
                                            max_length=self.max_len, truncation=True)['input_ids']} 


    
######## REPL ###############################################################################

class EmberEvalDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df: pd.DataFrame, 
                 data_col: str = 'merged_all',
                 indexed = False, 
                 left_id = "left_id",
                 right_id = "right_id"):
        self.df = df.copy()[data_col]
        self.n_samples = len(df)
        self.indexed = indexed
        self.left_id = left_id
        self.right_id = right_id
        if indexed:
            self.index = list(df.index)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.indexed:
            idx = self.index[idx]
        return self.df[idx]
                 
class EmberTripletDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 n_samples: int,
                 data_col: str = 'merged_all', 
                 supervision: pd.DataFrame = None,
                 params = None, 
                 left_id = "left_id",
                 right_id = "right_id"): #meant to pass in misc domain specific params
        self.df_l = df_l.copy()[data_col]
        self.df_r = df_r.copy()[data_col]
        self.n_samples = n_samples
        self.params = params
        self.left_id = left_id
        self.right_id = right_id
        self.triplets = self.gen_triplets(supervision)

    
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        pass

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        a, p, n = self.triplets[idx]
        #print(idx, a, p, n)
        return self.df_l[a], self.df_r[p], self.df_r[n]


class DemoDataset(EmberTripletDataset):
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = self.left_id #TODO: Check me
        q_col = self.right_id #TODO: Check me
        
        supervision, self.n_samples = update_supervision(supervision, self.n_samples, frac_pos=self.params['pos_frac'], skip=1)
        while len(triples) < self.n_samples:
            for _, record in supervision.iterrows():
                random_negative = np.random.choice(self.df_r.index)
                sampled_positive = np.random.choice(record[q_col])
                if random_negative not in record[q_col]:
                    triples.add((record[a_col], sampled_positive, random_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)
    
    
class DemoHardNegativeDataset(EmberTripletDataset):
    def update_supervision(self,
                           supervision: pd.DataFrame, 
                           n_samples: int,
                           frac_pos: float,
                           skip: int = 1):
    
        if frac_pos < 1:
            index = supervision.index[::skip]
            sample = np.random.choice(index, size=int(frac_pos*len(index)), replace=False)
            if skip > 1:
                sample = np.hstack([sample + i for i in range(skip)])
            output = supervision.loc[sample].copy()
            n_samples = len(output)
            return output, n_samples
        return supervision, n_samples
    
    def gen_triplets(self, 
                     supervision: pd.DataFrame):
        triples = set()
        a_col = self.left_id #TODO: Check me
        q_col = self.right_id #TODO: Check me
        negatives = self.params['negatives']
        
        supervision, self.n_samples = self.update_supervision(supervision, 
                                                              self.n_samples, 
                                                              frac_pos=self.params['pos_frac'], 
                                                              skip=1)
        while len(triples) < self.n_samples:
            for idx, record in supervision.iterrows():
                negative_list = negatives.loc[idx][q_col]
                sampled_negative = np.random.choice(negative_list)
                sampled_positive = np.random.choice(record[q_col])
                if sampled_negative not in record[q_col]:
                    triples.add((record[a_col], sampled_positive, sampled_negative))
                if len(triples) >= self.n_samples:
                    return list(triples)
        return list(triples)
    
    


