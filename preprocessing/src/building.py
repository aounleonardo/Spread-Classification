from dataclasses import asdict, fields
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import dgl
import networkx as nx

from spread_classification.utils import Followers

from .features import FEATURES_LIST, get_tweet_features

Node = Tuple[str, Optional[Dict[str, Any]]]
Edge = Tuple[str, str]
Spread = Tuple[List[Node], List[Edge]]


def _get_most_popular(nodes: List[Node]) -> Node:
    filtered = [tweet for id_, tweet in nodes if tweet is not None]
    if not filtered:
        return nodes[0]
    most_followed = max(*filtered, key=lambda tweet: tweet["user"]["followers_count"])
    return most_followed["id_str"], most_followed


def connect_retweets(
    tweet_id: str,
    retweets_ids: List[str],
    tweets: Dict[str, Dict[str, Any]],
    followers: Followers,
) -> Spread:
    nodes: List[Node] = [(tweet_id, tweets.get(tweet_id))] + [
        (id_, tweets.get(id_)) for id_ in retweets_ids
    ]
    edges: List[Edge] = []

    most_popular_tweet_id, most_popular_tweet = nodes[0]

    usernames: Set[str] = set(
        tweet["user"]["screen_name"] for _, tweet in nodes if tweet is not None
    )
    followers_subset = followers.subset(usernames)
    for index, (retweet_id, retweet) in enumerate(nodes[1:], start=1):
        retweet_username: str = retweet["user"]["screen_name"]
        for (candidate_id, candidate_tweet) in nodes[index - 1 :: -1]:
            if (
                retweet_username
                in followers_subset[candidate_tweet["user"]["screen_name"]]
            ):
                edges.append((candidate_id, retweet_id))
                break
        else:
            edges.append((most_popular_tweet_id, retweet_id))

        most_popular_tweet_id, most_popular_tweet = _get_most_popular(
            [(most_popular_tweet_id, most_popular_tweet), (retweet_id, retweet)]
        )
    return nodes, edges


def get_nodes_features(
    nodes: List[Node], edges: List[Edge]
) -> List[Tuple[str, Dict[str, Union[int, float]]]]:
    nodes_map = {tid: tweet for tid, tweet in nodes}
    parents = {edge[1]: edge[0] for edge in edges}
    return [
        (
            node_id,
            asdict(get_tweet_features(tweet, nodes_map[parents.get(node_id, node_id)])),
        )
        for node_id, tweet in nodes
    ]


def get_dgl_graph(nodes: List[Node], edges: List[Edge]):
    nx_graph = nx.DiGraph()
    nodes_features = get_nodes_features(nodes, edges)
    nx_graph.add_nodes_from(nodes_features)
    nx_graph.add_edges_from(edges)

    dgl_graph = dgl.DGLGraph()
    dgl_graph.from_networkx(nx_graph, node_attrs=FEATURES_LIST, edge_attrs=[])
    return dgl_graph
