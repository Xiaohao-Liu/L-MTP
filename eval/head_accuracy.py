import os
import torch
import json
from contextlib import contextmanager
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn.functional as F
from fastchat.model.model_adapter import get_conversation_template
from tqdm import tqdm
import argparse

from models.utils import reset_past_key_values, reset_tree_mode
from models.base.kv_cache import initialize_past_key_values

from .gen_results import load_accuracy_table, explore_graph

def get_accuracies(medusa, logit):
    seq_len, choices, topk = medusa.shape
    results = []
    for choice in range(choices):
        results.append(medusa[:-choice - 1,choice].eq(logit[choice + 1:,0]))
    return results

def get_choices(save_path, max_depth, max_child, num_iterations, num_heads):
    accuracies = load_accuracy_table(save_path)[:num_heads]
    
    for k in [1,5,10,20,40]:
        print(f"Top@{k}", accuracies[:,:k].sum(dim=-1))
    if len(accuracies) == 1:
        accept_nodes = [tuple([i]) for i in range(max_child[0])]
    else:
        accept_nodes = explore_graph(accuracies, max_depth, max_child, num_iterations)
    
    print("Accepted Nodes:", accept_nodes)
    return accuracies, accept_nodes
    

def main(
    model=None,
    data_path = "data/alpaca_eval.json",
    steps = 20,
    max_depth = 10,
    max_child = None,
    num_iterations = 100,
    force=False,
    num_heads = 3,
    suffix="",
    mtp_type="lmtp1",
):
    base_model = model.config.model_name
    n_head = model.config.n_head
    head_num_layers = model.config.head_num_layers
    os.makedirs("results/heads", exist_ok=True)
    save_path = os.path.join("results/heads", f"{mtp_type}_{suffix}_{base_model.replace('/', '_')}_{data_path.split('/')[-1].replace('.json','')}_heads_accuracy.pt")
    print(save_path)
    if not force and os.path.exists(save_path):
        return get_choices(save_path, max_depth, max_child, num_iterations, num_heads)
    
    tokenizer = model.get_tokenizer()
    
    data = json.load(open(data_path))
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.model)
    model.model.past_key_values = past_key_values
    model.model.past_key_values_data = past_key_values_data
    model.model.current_length_data = current_length_data
    model.model.past_hidden_states = None
    results = None

    for idx, sample in tqdm(enumerate(data)):
        conv = get_conversation_template("vicuna")
        conv.messages = []
        try:
            conv.append_message(conv.roles[0], sample["instruction"])
            conv.append_message(conv.roles[1], "")
        except:
            conv.append_message(conv.roles[0], sample[0]["content"])
            conv.append_message(conv.roles[1], "")
                
        prompt = conv.get_prompt()
        steps = steps
        logits_ids = []
        medusa_topk_ids = []

        with torch.inference_mode():
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).to(model.device)
            model.model.current_length_data.zero_() # this is for rerun
            reset_tree_mode(model)
            medusa_logits, outputs, logits = model(
                input_ids, past_key_values=past_key_values, output_orig=True
            )
            
            _, medusa_topk = medusa_logits[...,-1,:].topk(20, dim=-1)
            input_id = logits[:, -1:].argmax(dim=-1)
            logits_ids.append(input_id.detach().cpu())
            medusa_topk_ids.append(medusa_topk.detach().cpu())
                        
            for _ in range(steps):
                medusa_logits, outputs, logits = model(
                    input_id, past_key_values=past_key_values, output_orig=True
                )
                _, medusa_topk = medusa_logits[...,-1,:].topk(20, dim=-1)
                input_id = logits[:, -1:].argmax(dim=-1)
                logits_ids.append(input_id.detach().cpu())
                medusa_topk_ids.append(medusa_topk.detach().cpu())
            logits_ids = torch.stack(logits_ids, dim=0)
            medusa_topk_ids = torch.stack(medusa_topk_ids, dim=0).squeeze(2)
            if results is None:
                results = get_accuracies(medusa_topk_ids, logits_ids)
            else:
                cur_results = get_accuracies(medusa_topk_ids, logits_ids)
                for i in range(len(results)):
                    results[i] = torch.cat((results[i], cur_results[i]), dim=0)

    torch.save(results, save_path)
    
    accuracies, accept_nodes = get_choices(save_path, max_depth, max_child, num_iterations, num_heads)
    
    return accuracies, accept_nodes

    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    
