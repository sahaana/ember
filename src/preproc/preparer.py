import sys
import pandas as pd
from typing import Callable, List, Dict, Tuple, Sequence, NewType

from src.utils.preprocessing_utils import merge_columns

class Preparer():
    def __init__(self, 
                 df_l: pd.DataFrame, 
                 df_r: pd.DataFrame,
                 df_l_fname: str,
                 df_r_fname: str,
                 new_col_name: str = 'merged_all'):
        self.df_l = df_l.copy()
        self.df_r = df_r.copy()
        self.new_col_name = new_col_name
        self.df_l_fname = df_l_fname
        self.df_r_fname = df_r_fname

    def prepare(self, save: bool) -> Tuple[pd.DataFrame]:
        pass
    
    
class SentencePreparer(Preparer):
    def prepare(self, save = True) -> Tuple[pd.DataFrame]:
        def merge_all(df:pd.DataFrame, new_col = self.new_col_name, separator = "[SEP]") -> pd.DataFrame:
            df[new_col] = ''
            for col in list(df.columns):
                df[new_col] =  df[new_col] + f" {separator} " + f" {col} " + df[col].astype(str)
            return df
        
        df_l_processed = merge_all(self.df_l)
        df_r_processed = merge_all(self.df_r)
        
        if save:
            df_l_processed.to_pickle(self.df_l_fname)
            df_r_processed.to_pickle(self.df_r_fname) 
        
        return(df_l_processed, df_r_processed)