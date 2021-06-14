from transformers import AutoTokenizer, DistilBertTokenizer
from typing import Callable, List, Dict, Tuple, Sequence, NewType
import torch
import os
from datetime import datetime


def save_torch_model(path, model):
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = path + datetime.now().strftime("%H-%M-%d-%m-%y")
    torch.save(model.state_dict(), filepath)
    print(f"Saved Model: {filepath}")
    return filepath

    
def to_cuda(tensor_list, levels):
    if levels == 0:
        return [i.cuda() for i in tensor_list]
    else:
        return [to_cuda(i, levels - 1) for i in tensor_list]


def tokenize_batch(inputs: List,
                   tokenizer: DistilBertTokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased'),
                   max_length: int = 512) -> torch.Tensor:
    outputs = []
    masks = []
    for i in inputs: # iterating a,p,n
        out = tokenizer(list(i), return_tensors='pt', padding=True,
                        max_length=max_length, truncation=True)
        outputs.append(out['input_ids'])
        masks.append(out['attention_mask'])
    return outputs, masks

def tokenize_single_data_batch(inputs: List,
                               tokenizer: DistilBertTokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased'),
                               max_length: int = 512) -> torch.Tensor:
    out = tokenizer(list(inputs), return_tensors='pt', padding=True,
                        max_length=max_length, truncation=True)
    return out['input_ids'], out['attention_mask']
