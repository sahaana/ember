import sys
import argparse
import pandas as pd
from types import SimpleNamespace   
from typing import Callable, List, Dict, Tuple, Sequence, NewType

from src.preproc.preparer import SentencePreparer
from src.preproc.pretrainer import BM25Pretrainer, train_MLM, compute_bm25_negatives
from src.replearn.encoder import train_embedding
from src.postproc.retriever import perform_knn
from src.utils.file_utils import load_config

def run_ember(config): 
    
    # Load data
    conf = SimpleNamespace(**config)
    df_l = pd.read_csv(f"{conf.data_path}/left.csv", index_col=conf.ID_left)
    df_r = pd.read_csv(f"{conf.data_path}/right.csv", index_col=conf.ID_right)
    train_supervision = pd.read_pickle(f"{conf.data_path}/train.pkl")
    test_supervision = pd.read_pickle(f"{conf.data_path}/test.pkl")
    print("Loaded Data")
    
    # If LEFT join, keep as is. If RIGHT join, just switch the datasets and fields. 
    if conf.join_type == "LEFT":
        pass
    elif conf.join_type == "RIGHT":
        tmp = df_l.copy()
        df_l = df_r.copy()
        df_r = tmp
        
        tmp = conf.ID_left
        conf.ID_left = conf.ID_right
        conf.ID_right = tmp
    else:
        print("Unsupported Join Type")
        sys.exit()
        
    # Set parameters 
    conf.train_size = int(len(train_supervision)*conf.train_frac)
    conf.bert_path = f'{conf.data_path}/models/MLM/{conf.mlm_model_name}'
    df_l_fprocname = f"{conf.data_path}/left_processed.pkl"
    df_r_fprocname = f"{conf.data_path}/right_processed.pkl"
    
    # Perform Ember Pipeline from preparing to retrieval
    preparer = SentencePreparer(df_l,
                                df_r, 
                                df_l_fprocname,
                                df_r_fprocname,
                                conf.new_col_name)
    df_l_proc, df_r_proc = preparer.prepare()
    print("Prepared Data")
                                
    BM25_indices = BM25Pretrainer(df_l_proc, df_r_proc, conf)
    print("Computed BM25")
    
    train_MLM(df_l_proc, df_r_proc, BM25_indices, conf)
    print("Pretrained MLM")
    
    train_negatives = compute_bm25_negatives(train_supervision, BM25_indices, conf)
    print("Processed BM25 Hard Negatives")
    
    trained_model = train_embedding(df_l_proc, df_r_proc, train_supervision, train_negatives, conf)
    print(f"Learned Embedding: {trained_model}")
    
    results = perform_knn(df_l_proc, df_r_proc, test_supervision, trained_model, conf)
    print("Indexed")
          
    print(results)
    return results
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform Ember Default Pipeline')
    parser.add_argument('-c', "--config", required=True,
                        help="Config file")
    args = parser.parse_args()
    config = load_config(args.config)
    run_ember(config)
