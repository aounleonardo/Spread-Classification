import argparse
from typing import Dict, List, Set

from dgl.data.utils import save_graphs

from spread_classification.preprocessing import (
    Edge,
    Node,
    Spread,
    connect_retweets,
    get_dgl_graph,
    get_nodes_features,
)
from spread_classification.utils import (
    Followers,
    extract_tweets,
    get_parent_tweet_id,
    get_retweet_ids,
)


def unify_topic(cascades: Dict[str, Spread], tweets_ids: Set[str]) -> Spread:
    nodes: List[Node] = [("root", {"type": "root"})]
    edges: List[Edge] = [
        edge for _, cascade_edges in cascades.values() for edge in cascade_edges
    ]
    incoming_edges = set(destination for _, destination in edges)
    spreads = [
        (tweet_id, cascade)
        for tweet_id, cascade in cascades.items()
        if tweet_id not in incoming_edges
    ]

    for tweet_id, (spread_nodes, _) in spreads:
        nodes += spread_nodes
        _, tweet = spread_nodes[0]
        if tweet is None:
            edges.append(("root", tweet_id))
        else:
            parent_id = get_parent_tweet_id(tweet)
            if parent_id is None:
                edges.append(("root", tweet_id))
            elif parent_id in tweets_ids:
                edges.append((parent_id, tweet_id))
            else:
                nodes.append((parent_id, None))
                edges.append(("root", parent_id))
                edges.append((parent_id, tweet_id))
    return nodes, edges


def main(args):
    tweets = extract_tweets(args.tweets_dir, args.count)
    retweets_per_tweet = get_retweet_ids(tweets)

    cascades: Dict[str, Spread] = {
        tweet_id: connect_retweets(
            tweet_id, retweet_ids, tweets, Followers(args.followers_dir)
        )
        for (tweet_id, retweet_ids) in retweets_per_tweet.items()
    }
    nodes, edges = unify_topic(cascades, set(tweets.keys()))
    graph = get_dgl_graph(nodes, edges)

    save_graphs(args.graph_output.name, graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of tweets, and build a graph representing it."
    )
    parser.add_argument("--tweets-dir", "--tweets", "-i", type=str, required=True)
    parser.add_argument(
        "--graph-output", "-o", type=argparse.FileType("w"), required=True
    )
    parser.add_argument("--followers-dir", "-f", type=str, required=True)
    parser.add_argument("--count", "-c", type=int, nargs="?", default=None)

    main(parser.parse_args())
