import argparse
import os
from functools import reduce
from typing import Any, Collection, Dict, List, Set, Tuple

from dgl.data.utils import save_graphs

from spread_classification.preprocessing import (
    Edge,
    Node,
    Spread,
    connect_retweets,
    get_dgl_graph,
)
from spread_classification.utils import (
    Followers,
    extract_tweets,
    get_children_ids,
    get_retweet_ids,
)


def get_cascade(
    node_id: str,
    tweets: Dict[str, Dict[str, Any]],
    children: Dict[str, List[str]],
    retweets: Dict[str, List[str]],
    followers: Followers,
) -> Spread:
    node = (node_id, tweets.get(node_id))
    if not children.get(node_id, []):
        return ([node], [])

    children_cascades = {
        child_id: get_cascade(child_id, tweets, children, retweets, followers)
        for child_id in children[node_id]
    }
    all_children_nodes, all_children_edges = zip(*children_cascades.values())
    nodes = [node] + reduce(list.__add__, all_children_nodes)
    _, connections = connect_retweets(
        node_id, retweets.get(node_id, []), tweets, followers
    )
    edges = reduce(list.__add__, all_children_edges) + connections
    return (nodes, edges)


def main(args):
    output_prefix = os.path.split(
        os.path.split(os.path.dirname(args.tweets_dir + "/"))[0]
    )[-1]
    followers = Followers(args.followers_dir)

    tweets = extract_tweets(args.tweets_dir, args.count)
    retweets_per_tweet = get_retweet_ids(tweets)

    children_per_tweet = get_children_ids(tweets)
    children_ids = set(
        child_id for children in children_per_tweet.values() for child_id in children
    )
    head_ids = set(children_per_tweet.keys()) - children_ids

    cascades = {
        head_id: get_cascade(
            head_id, tweets, children_per_tweet, retweets_per_tweet, followers
        )
        for head_id in head_ids
    }

    for head_id, (nodes, edges) in cascades.items():
        graph = get_dgl_graph(
            [("root", {"type": "root"})] + nodes, [("root", head_id)] + edges
        )
        save_graphs(
            os.path.join(args.cascades_output, f"{output_prefix}_{head_id}.bin"), graph,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of tweets, and build all cascades representing it."
    )
    parser.add_argument("--tweets-dir", "--tweets", "-i", type=str, required=True)
    parser.add_argument("--cascades-output", "-o", type=str, required=True)
    parser.add_argument("--followers-dir", "-f", type=str, required=True)
    parser.add_argument("--count", "-c", type=int, nargs="?", default=None)

    main(parser.parse_args())
