import os
import argparse
from typing import Set, Optional
from functools import reduce
import json

from spread_classification.utils import (
    Followers,
    extract_tweets,
    fetch_using_twint,
    get_followers_config,
    get_retweet_ids,
)

from spread_classification.fetch.fakenewsnet.util.util import Config, DataCollector


def process_choice(collector, choice):
    all_tweets_count = hydrated_count = 0

    tweets_dir = (
        f"{collector.config.dump_location}/{choice['news_source']}/{choice['label']}"
    )
    news_list = collector.load_news_file(choice)
    for news in news_list:
        tweet_ids = news.tweet_ids
        all_tweets_count += len(tweet_ids)
        filtered_ids = [
            id_
            for id_ in tweet_ids
            if os.path.exists(f"{tweets_dir}/{news.news_id}/tweets/{id_}.json")
        ]
        hydrated_count += len(filtered_ids)
    print(choice, hydrated_count, all_tweets_count)


def main(args):
    json_object = json.load(args.config)
    config = Config(
        json_object["dataset_dir"],
        json_object["dump_location"],
        json_object["tweet_keys_file"],
        int(json_object["num_process"]),
    )
    data_collector = DataCollector(config)
    data_choices = json_object["data_collection_choice"]
    for choice in data_choices:
        process_choice(data_collector, choice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a Sysomos collection of tweets, fetch the "
        + "needed followers list, and build a graph representing it."
    )
    parser.add_argument("--config", "-c", type=argparse.FileType("r"), required=True)
    main(parser.parse_args())
