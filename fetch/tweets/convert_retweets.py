import argparse
import os
import json
import shutil
from typing import List, Dict, Any


def get_retweets(dir_: str) -> List[Dict[str, Any]]:
    retweets_filenames = os.listdir(dir_)
    retweets = []
    for filename in retweets_filenames:
        filepath = os.path.join(dir_, filename)
        with open(filepath) as file:
            try:
                retweets += json.load(file).get("retweets", [])
            except Exception as e:
                print(e, filepath)

    return retweets


def merge_tweets_and_retweets(
    folder_name: str, source_dir: str, destination_dir: str
) -> None:
    tweets_dir = os.path.join(source_dir, folder_name, "tweets")
    dump_dir = os.path.join(destination_dir, folder_name, "tweets")
    if not os.path.exists(tweets_dir) or os.path.exists(dump_dir):
        return
    shutil.copytree(tweets_dir, dump_dir)

    retweets_dir = os.path.join(source_dir, folder_name, "retweets")
    if not os.path.exists(retweets_dir):
        return
    retweets = get_retweets(retweets_dir)
    for retweet in retweets:
        id_ = retweet["id_str"]
        with open(os.path.join(dump_dir, f"{id_}.json"), "w") as retweet_file:
            json.dump(retweet, retweet_file)


def main(args):
    ratings = os.listdir(args.dataset_dir)
    for rating in ratings:
        source_dir = os.path.join(args.dataset_dir, rating)
        output_dir = os.path.join(args.output_dir, rating)
        folders = os.listdir(source_dir)
        for folder in folders:
            merge_tweets_and_retweets(folder, source_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine subfolders of tweets and retweets of a list of folders to not consider retweets as a "
        + "special case"
    )
    parser.add_argument("--dataset-dir", "-i", type=str, required=True)
    parser.add_argument("--output-dir", "-o", type=str, required=True)

    main(parser.parse_args())
