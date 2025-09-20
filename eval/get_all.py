
from .head_accuracy import main as get_head_accepts
import os
from models.lmtp.model import LMTPModel
from models.medusa.model import MedusaModel
import json
import torch
import matplotlib.pyplot as plt
import copy
import networkx as nx
import torch
import argparse

def eval(mtp_type, model_name_or_path, stage1_pretrained_path, stage2_pretrained_path=None, num_head=3):
    if stage1_pretrained_path is None:
        stage1_pretrained_path = model_name_or_path
    checkpoint_path = stage1_pretrained_path if stage2_pretrained_path is None else stage2_pretrained_path
    from models import get_mtp_model_inference
    
    model = get_mtp_model_inference(
        mtp_type,
        model_name_or_path,
        tokenizer=None,
        model=None,
        train_mode = False,
        stage1_pretrained_path=stage1_pretrained_path,
        stage2_pretrained_path=stage2_pretrained_path,
        num_head=num_head,
    ).to("cuda")
    
    model_name = model.config.model_name
    times = 2 if "lmtp" in mtp_type else 1 # lmtp has 2 times more horizon than mtp
    model.to("cuda")
    model.stage = 1 if stage2_pretrained_path is None else 2
    accuracies, accept_nodes = get_head_accepts(
        model, suffix=f"{checkpoint_path.replace('/', '_')}", max_child=[10,10,10,10,10,10,10,10][:num_head*times+1], max_depth=num_head*times, num_heads = num_head*times, mtp_type=mtp_type
    )
    
    del model
    torch.cuda.empty_cache()

    return accuracies, accept_nodes, model_name

def main(
    mtp_type="lmtp",
    model_name_or_path="Qwen/Qwen2-7B-Instruct",
    stage1_pretrained_path=None,
    stage2_pretrained_path=None,
    num_head=3,
):
    results = {}
    accuracies, accept_nodes, model_name = eval(mtp_type, model_name_or_path, stage1_pretrained_path, stage2_pretrained_path, num_head)
    results = {
        "accept_nodes": str(accept_nodes), 
        "head_accuracy": {
            "top1": accuracies[:,:1].sum(dim=-1).cpu().tolist(),
            "top5": accuracies[:,:5].sum(dim=-1).cpu().tolist(),
            "top10": accuracies[:,:10].sum(dim=-1).cpu().tolist(),
            "top20": accuracies[:,:20].sum(dim=-1).cpu().tolist(),
            },
    }
    
    if os.path.exists("heads_accuracy.json"):
        with open("heads_accuracy.json", "r") as f:
            heads_accuracy = json.load(f)
    else:
        heads_accuracy = {}
        
    with open("heads_accuracy.json", "w") as f:
        if mtp_type not in heads_accuracy:
            heads_accuracy[mtp_type] = {}
        if model_name not in heads_accuracy[mtp_type]:
            heads_accuracy[mtp_type][model_name] = {}
        if num_head not in heads_accuracy[mtp_type][model_name]:
            heads_accuracy[mtp_type][model_name][num_head] = {}
            
        heads_accuracy[mtp_type][model_name][num_head] = {
            "results": results,
        }
        f.write(json.dumps(heads_accuracy))
    
    
if __name__ == "__main__":
    from fire import Fire
    Fire(main)


