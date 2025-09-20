
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
    times = 2 if "lmtp" in mtp_type else 1
    model.to("cuda")
    model.stage = 1 if stage2_pretrained_path is None else 2
    accuracies, accept_nodes = get_head_accepts(
        model, suffix=f"{checkpoint_path.replace('/', '_')}", max_child=[10,10,10,10,10,10,10,10][:num_head*times+1], max_depth=num_head*times, num_heads = num_head*times
    )
    
    del model
    torch.cuda.empty_cache()

    return accuracies, accept_nodes, model_name

def plot_and_save_graph(accept_nodes, output_path):
    plt.figure(figsize=(40, 20)) 

    G = nx.DiGraph()

    for path in accept_nodes:
        for i in range(len(path)):
            if i == 0:
                parent = 'root'
            else:
                parent = tuple(path[:i])
            child = tuple(path[:i+1])
            G.add_edge(parent, child)

    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", font_size=10, width=2, edge_color="gray")
    plt.savefig(output_path)
    
def main(
    mtp_type="lmtp",
    model_name_or_path="Qwen/Qwen2-7B-Instruct",
    stage1_pretrained_path=None,
    stage2_pretrained_path=None,
    num_head=3,
):
    results = {}
    accuracies, accept_nodes, model_name = eval(mtp_type, model_name_or_path, stage1_pretrained_path, stage2_pretrained_path, num_head)
    
    plot_and_save_graph(accept_nodes, f"results/tree/{model_name}_{mtp_type}_tree.png")
    
if __name__ == "__main__":
    from fire import Fire
    Fire(main)
