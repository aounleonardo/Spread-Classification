import networkx as nx
import dgl
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple

COLOUR_LIST: List[str] = [
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#00ffff",
    "#ff00ff",
    "#ffff00",
]


def _get_nodelists(
    graph: nx.DiGraph, colour_feature: str, size_feature: str, size_order: int
) -> Tuple[List[int], List[str], List[int]]:
    nodelist = graph.nodes()
    colours = [COLOUR_LIST[nodelist[node][colour_feature]] for node in nodelist]
    sizes = [
        max(10, min(100, nodelist[node][size_feature].item() // size_order)) for node in nodelist
    ]
    return [id_ for id_ in nodelist], colours, sizes


def draw(graph: dgl.DGLGraph) -> matplotlib.figure.Figure:
    figure = plt.figure()
    nxg = graph.to_networkx(node_attrs=["type_", "followers_count"])
    nodelist, colours, sizes = _get_nodelists(
        nxg, colour_feature="type_", size_feature="followers_count", size_order=100
    )
    pos = nx.kamada_kawai_layout(nxg)
    nx.draw_networkx_nodes(nxg, nodelist=nodelist, pos=pos, node_size=sizes, node_color=colours)
    nx.draw_networkx_edges(nxg, edge_list=nxg.edges(), pos=pos, width=.1, arrowsize=5)

    return figure
