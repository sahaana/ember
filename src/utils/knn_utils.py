from typing import Callable, List, Dict, Tuple, Sequence, NewType
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support


## Processing ############################################################################################


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.k = k

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def kneighbors(self, X):
        return self.index.search(X.astype(np.float32), k=self.k)
    
def create_neib_mask(num_data, num_neib):
    return np.reshape([range(num_data)]*num_neib, (num_neib, num_data)).T

def compute_top_k_pd(routine, params, k_max, thresh = None):
    knn_results = defaultdict(list)
    for k in range(1, k_max+1):
        ret_avg, ret_count, all_avg, all_count, MRR, results, all_results, MRR_results = routine(*params, k=k, thresh=thresh)
        print(f"k: {k} \t ret avg: {ret_avg} \t ret count: {ret_count} \t all avg: {all_avg} \t all count: {all_count} \t MRR: {MRR}")
        knn_results['k'].append(k)
        knn_results['ret_avg'].append(ret_avg)
        knn_results['ret_count'].append(ret_count)
        knn_results['all_avg'].append(all_avg)
        knn_results['all_count'].append(all_count)
        knn_results['MRR'].append(MRR)

    knn_results = pd.DataFrame(knn_results)
    return knn_results


def process_results(results: List[pd.DataFrame]) -> Tuple[int, pd.DataFrame]:
    combined = pd.concat(results, axis=1)
    if len(results) > 1:
        averaged = pd.DataFrame({'k' : combined['k'].mean(axis=1), 
                                 'ret_avg': combined['ret_avg'].mean(axis=1), 
                                 'all_avg': combined['all_avg'].mean(axis=1),
                                 'MRR_avg': combined['MRR'].mean(axis=1),
                                 'ret_count': combined['ret_count'].mean(axis=1),
                                 'all_count': combined['all_count'].mean(axis=1),
                                }).set_index('k')
    else:
        averaged = pd.DataFrame({'k' : combined['k'],
                                 'ret_avg': combined['ret_avg'],
                                 'all_avg': combined['all_avg'],
                                 'MRR_avg': combined['MRR'],
                                 'ret_count': combined['ret_count'],
                                 'all_count': combined['all_count'],
                                }).set_index('k')
    return len(results), averaged


def return_metrics(res: pd.DataFrame, k_list: List[int], thresh_list: List[float]) -> Tuple[pd.DataFrame, pd.DataFrame] :
    k_values = res.loc[k_list].copy()
    thresh_values = pd.DataFrame({'thresh': thresh_list, 
                                  "k_one": [res[res['ret_avg'] >= i].index[0] if len(res[res['ret_avg'] >= i]) > 0 else 300 for i in thresh_list],
                                  "k_all": [res[res['all_avg'] >= i].index[0] if len(res[res['all_avg'] >= i]) > 0 else 300 for i in thresh_list]}).set_index('thresh')
    return k_values, thresh_values


## KNN Modules ############################################################################################

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)


def knn_metrics(dists: np.array,
                neibs: np.array,
                supervision: pd.DataFrame,
                left_indexing: np.array,
                right_indexing: np.array,
                l_id: str,
                r_id: str,
                k: int = None,
                thresh: float = None):
    supervision = supervision.set_index(l_id)
    mode = r_id
    if k is not None:
        neibs = right_indexing[neibs[:,:k]]
    else:
        pass # TODO
    results = []
    MRR_results = []
    all_captured = []
    
    for idx, row in enumerate(neibs):
        matches = 0
        mrr_count = 0
        first_relevant = np.inf
        
        qid = left_indexing[idx]
        
        if qid in supervision.index:
            true_matches = supervision.loc[qid][mode]
            true_matches = set(true_matches)
            for entry in row: 
                mrr_count += 1.
                if entry in true_matches:
                    first_relevant = min(mrr_count, first_relevant)
                    matches += 1
                    
            all_matches = 1 if len(true_matches) == matches else 0
            one_match = 1 if matches > 0 else 0
            
            results.append(one_match)
            all_captured.append(all_matches)
            MRR_results.append(1./first_relevant)
    return np.mean(results), np.sum(results), np.mean(all_captured), np.sum(all_captured), \
           np.mean(MRR_results), results, all_captured, MRR_results  
