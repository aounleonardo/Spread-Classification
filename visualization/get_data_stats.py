import argparse
import itertools as it
import json
import math
from functools import reduce
from multiprocessing import Pool

from spread_classification.fetch.tweets.prune_retweet_tree import prune_small_branches
from spread_classification.preprocessing.build_cascades import get_cascades
from spread_classification.preprocessing.build_spreads import get_spreads
from spread_classification.utils import (
    Followers,
    UrlsDataset,
    extract_tweets,
    get_children_ids,
    get_retweet_ids,
)


def get_prune_lengths(settings):
    return range(2, 1 + settings["largest_prune"])


def check_needed_followers(cascade, followers, max_follower_count=5000):
    needed = [
        node[1]["user"]["screen_name"]
        for node in cascade[0][:-1]
        if node[1]["user"]["followers_count"] <= max_follower_count
    ]
    available = [author for author in needed if author in followers]
    return len(needed), len(available)


def process_url(path, followers, settings):
    tweets = extract_tweets(path)
    retweets = get_retweet_ids(tweets)
    children = get_children_ids(tweets)

    cascades = get_cascades(tweets, followers)
    spreads = get_spreads(tweets, followers)

    prune_lengths = get_prune_lengths(settings)
    pruned = {
        length: prune_small_branches(tweets, min_branch_size=length)
        for length in prune_lengths
    }

    needed_followers, available_followers = (
        sum(values)
        for values in zip(
            *[
                check_needed_followers(cascade, followers)
                for cascade in cascades.values()
            ]
        )
    )

    return {
        "full_size": len(tweets),
        **{f"pruned_{length}_size": len(pruned[length]) for length in prune_lengths},
        "needed_followers": needed_followers,
        "available_followers": available_followers,
        "cascades": {k: len(cascade[0]) for (k, cascade) in cascades.items()},
        "spreads": {k: len(spreads[0]) for (k, spreads) in spreads.items()},
    }


def combine_processed(processed, settings):
    prune_lengths = get_prune_lengths(settings)
    fun = lambda acc, new: {
        "full_size": acc["full_size"] + [new["full_size"]],
        **{
            f"pruned_{length}_size": acc[f"pruned_{length}_size"]
            + [new[f"pruned_{length}_size"]]
            for length in prune_lengths
        },
        "needed_followers": acc["needed_followers"] + [new["needed_followers"]],
        "available_followers": acc["available_followers"]
        + [new["available_followers"]],
        "cascades": {**acc["cascades"], **new["cascades"]},
        "spreads": {**acc["spreads"], **new["spreads"]},
    }
    return reduce(
        fun,
        processed,
        {
            "full_size": [],
            **{f"pruned_{length}_size": [] for length in prune_lengths},
            "needed_followers": [],
            "available_followers": [],
            "cascades": {},
            "spreads": {},
        },
    )


def main(args):
    dataset = UrlsDataset(args.urls)
    followers = Followers(args.followers)

    pool = Pool(processes=args.processes)
    ids = (dataset[id_] for id_ in dataset.labels.keys())
    followers_gen = it.repeat(followers)
    settings = it.repeat({"largest_prune": args.largest_prune})
    processed = pool.starmap(process_url, zip(ids, followers_gen, settings))
    combined = combine_processed(processed, next(settings))
    json.dump(combined, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", "-i", type=str, required=True)
    parser.add_argument("--followers", "-f", type=str, required=True)
    parser.add_argument("--largest-prune", "--len", type=int, default=6)
    parser.add_argument("--processes", "-p", type=int, default=4)
    parser.add_argument("--output", "-o", type=argparse.FileType("w"), required=True)
    main(parser.parse_args())
