import argparse
import os
from functools import reduce
from typing import Any, Collection, Dict, List, Set, Tuple

import torch
from dgl.data.utils import save_graphs

from spread_classification.modeling import TextClassifier
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


def get_cascades(
    tweets: Dict[str, Dict[str, Any]], followers: Followers
) -> Dict[str, Spread]:
    retweets_per_tweet = get_retweet_ids(tweets)
    retweet_ids = set(
        retweet_id
        for retweets in retweets_per_tweet.values()
        for retweet_id in retweets
    )
    head_ids = set(retweets_per_tweet.keys()) - retweet_ids

    return {
        head_id: connect_retweets(
            head_id, retweets_per_tweet.get(head_id, []), tweets, followers
        )
        for head_id in head_ids
    }


def main(args):
    output_prefix = os.path.split(
        os.path.split(os.path.dirname(args.tweets_dir + "/"))[0]
    )[-1]

    cascades = get_cascades(
        extract_tweets(args.tweets_dir, args.count), Followers(args.followers_dir)
    )

    text_encoder = (
        TextClassifier.from_file(args.text_encoder_path, torch.device("cuda:0"))
        if args.text_encoder_path is not None
        else None
    )

    for head_id, (nodes, edges) in cascades.items():
        if len(nodes) >= args.min_cascade_length:
            graph = get_dgl_graph(
                [("root", {"type": "root"})] + nodes,
                [("root", head_id)] + edges,
                text_encoder=text_encoder,
            )
            save_graphs(
                os.path.join(args.cascades_output, f"{output_prefix}_{head_id}.bin"),
                graph,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of tweets, and build all cascades representing it."
    )
    parser.add_argument("--tweets-dir", "--tweets", "-i", type=str, required=True)
    parser.add_argument("--cascades-output", "-o", type=str, required=True)
    parser.add_argument("--followers-dir", "-f", type=str, required=True)
    parser.add_argument("--min-cascade-length", "--len", type=int, default=-1)
    parser.add_argument("--count", "-c", type=int, nargs="?", default=None)
    parser.add_argument("--text-encoder-path", "-e", type=str, default=None)

    main(parser.parse_args())
