import argparse
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from spread_classification.utils import (
    extract_tweets,
    get_children_ids,
    get_parent_tweet_id,
)


def get_branch_size(
    node_id: str, children: Dict[str, List[str]], dp: Dict[str, int]
) -> int:
    if node_id not in children:
        return 1
    if node_id not in dp:
        children_lengths = [
            get_branch_size(child_id, children, dp) for child_id in children[node_id]
        ]
        dp[node_id] = sum(children_lengths) + 1
    return dp[node_id]


def get_unwanted_tweet_ids(
    tweets: Dict[str, Dict[str, Any]],
    children: Dict[str, List[str]],
    min_branch_size: int,
) -> Set[str]:
    children_ids = set(
        child_id for tweet_children in children.values() for child_id in tweet_children
    )

    branch_head_ids = set(children.keys()) - children_ids

    dp = {}
    unwanted_branches = set(
        branch
        for branch in branch_head_ids
        if get_branch_size(branch, children, dp) < min_branch_size
    )

    return set(
        tweet_id
        for head_id in unwanted_branches
        for tweet_id in [head_id] + children.get(head_id, [])
    )


def prune_small_branches(
    tweets: Dict[str, Dict[str, Any]], min_branch_size: int,
) -> Dict[str, Dict[str, Any]]:
    children_per_tweet = get_children_ids(tweets)
    unwanted_tweet_ids = get_unwanted_tweet_ids(
        tweets, children_per_tweet, min_branch_size
    )
    remaining_tweet_ids = set(tweets.keys()) - unwanted_tweet_ids
    return {
        tweet_id: tweets[tweet_id]
        for tweet_id in remaining_tweet_ids
        if tweet_id in tweets
    }


def main(args):
    tweets = extract_tweets(args.tweets_dir)

    remaining_tweets = prune_small_branches(tweets, args.min_branch_size)
    for tweet_id, tweet in remaining_tweets.items():
        with open(f"{args.pruned_output}/{tweet_id}.json", "w") as file:
            json.dump(tweet, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of tweets, and build a graph representing it."
    )
    parser.add_argument("--tweets-dir", "--tweets", "-i", type=str, required=True)
    parser.add_argument("--pruned-output", "-o", type=str, required=True)
    parser.add_argument("--min-branch-size", "--len", nargs="?", type=int, default=0)

    main(parser.parse_args())
