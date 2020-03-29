import argparse
import os
from functools import reduce
from typing import Any, Collection, Dict, List, Set, Tuple

from dgl.data.utils import save_graphs

from spread_classification.preprocessing import get_tweet_content
from spread_classification.utils import (
    extract_tweets,
    get_retweet_ids,
)


def main(args):
    output_prefix = os.path.split(
        os.path.split(os.path.dirname(args.tweets_dir + "/"))[0]
    )[-1]

    tweets = extract_tweets(args.tweets_dir, args.count)
    retweets_per_tweet = get_retweet_ids(tweets)
    retweet_ids = set(
        retweet_id
        for retweets in retweets_per_tweet.values()
        for retweet_id in retweets
    )
    head_ids = set(retweets_per_tweet.keys()) - retweet_ids

    for head_id in head_ids:
        head_retweets = retweets_per_tweet.get(head_id, [])
        story_tweets = [
            tweets[id_] for id_ in [head_id] + head_retweets if id_ in tweets
        ]
        if len(story_tweets) >= args.min_cascade_length:
            story = " [CLS] " + " [SEP] ".join(
                [get_tweet_content(tweet) for tweet in story_tweets]
            )
            with open(
                os.path.join(args.stories_output, f"{output_prefix}_{head_id}.txt"),
                "w",
            ) as file:
                file.write(story)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a folder of tweets, and build all cascades representing it."
    )
    parser.add_argument("--tweets-dir", "--tweets", "-i", type=str, required=True)
    parser.add_argument("--stories-output", "-o", type=str, required=True)
    parser.add_argument("--min-cascade-length", "--len", type=int, default=-1)
    parser.add_argument("--count", "-c", type=int, nargs="?", default=None)

    main(parser.parse_args())
