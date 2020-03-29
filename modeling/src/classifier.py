import dgl
import dgl.function as fn
import torch
import torch.nn.functional as F
from torch import nn

from spread_classification.preprocessing import FEATURES_LIST, NodeTypes, Devices
from torch.nn.functional import one_hot

# Sends a message of node feature h.
msg = fn.copy_src(src="h", out="m")


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(
        nodes.mailbox["m"], 1
    )  ## should think about this, even sum didnt solve it, was 'mean'
    return {"h": accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""

    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data["h"])
        h = self.activation(h)
        return {"h": h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, features):
        # Initialize the node features with h.
        g.ndata["h"] = features
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop("h")


def _get_graph_features(graph: dgl.DGLGraph) -> torch.Tensor:
    features = [
        # For undirected graphs, in_degree is the same as
        # out_degree. I changed it to out_degrees() because it's a spanning tree
        graph.out_degrees(),
        graph.ndata["retweet_count"],
        graph.ndata["favorite_count"],
        graph.ndata["diff_time"],
        graph.ndata["followers_count"],
        graph.ndata["friends_count"],
        graph.ndata["statuses_count"],
        graph.ndata["listed_count"],
        graph.ndata["favourites_count"],
        graph.ndata["twitter_age"],
        F.one_hot(graph.ndata["type_"], num_classes=len(NodeTypes)),
        F.one_hot(graph.ndata["device"], num_classes=len(Devices)),
        F.one_hot(graph.ndata["verified"], num_classes=2),
        F.one_hot(graph.ndata["protected"], num_classes=2),
    ]
    return torch.cat([feature.view(len(graph), -1).float() for feature in features], dim=-1)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.input_dimension = 10 + len(NodeTypes) + len(Devices) + 2 * 2

        gcn_hidden_dimension = 16
        fc_hidden_dimension = 8
        self.add_module('batch_norm', nn.BatchNorm1d(self.input_dimension))
        self.gcns = nn.ModuleList(
            [
                GCN(self.input_dimension, gcn_hidden_dimension, F.relu),
                GCN(gcn_hidden_dimension, gcn_hidden_dimension, F.relu),
            ]
        )
        self.fcs = nn.ModuleList(
            [
                nn.Linear(gcn_hidden_dimension, fc_hidden_dimension),
                nn.Linear(fc_hidden_dimension, 1),
            ]
        )

    def _device(self):
        return next(self.parameters()).device

    def forward(self, g):
        bg = dgl.batch(g)
        h = _get_graph_features(bg).to(self._device())
        # h = self.batch_norm(h)
        for gcn in self.gcns:
            h = gcn(bg, h)
        bg.ndata["h"] = h
        hg = dgl.mean_nodes(bg, "h")
        for fc in self.fcs:
            hg = fc(hg)
        return torch.sigmoid(hg).squeeze()
