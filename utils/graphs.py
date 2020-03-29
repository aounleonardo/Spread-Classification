import torch
import dgl
from dgl.data.utils import load_graphs
from dgl.graph import DGLGraph
from typing import List

def load_graph(filename: str) -> DGLGraph:
    graphs, _ = load_graphs(filename)
    return graphs[0]

def collate(samples: List[DGLGraph]):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
        
