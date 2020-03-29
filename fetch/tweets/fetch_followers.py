import argparse
from typing import Set, Optional
from functools import reduce

from spread_classification.utils import (
    Followers,
    extract_tweets,
    fetch_using_twint,
    get_followers_config,
    get_retweet_ids,
)


def _save_followers_content(followers, username, content):
    if content:
        followers[username] = content.splitlines()
    else:
        print(f"Content for {username} is empty")


def fetch_followers_of_authors(
    usernames, limit, retries, followers, force, threads=None
):
    if not force:
        usernames = [name for name in usernames if name not in followers]
    twint_configs = [get_followers_config(name, limit) for name in usernames]
    print(f"Launching {len(twint_configs)} twint configs on {threads} threads")
    _ = fetch_using_twint(
        twint_configs,
        retries,
        threads=threads,
        callback=lambda username, content: _save_followers_content(
            followers.clone(), username, content
        ),
    )


def get_authors_to_process(tweets, retweets_map, max_followers=5000):
    retweets_to_process = (
        tweets.get(retweet_id)
        for retweet_ids in retweets_map.values()
        for retweet_id in retweet_ids[:-1]
    )
    return set(
        retweet.get("user").get("screen_name")
        for retweet in retweets_to_process
        if retweet is not None
    )


def get_topic_authors(tweets_dir: str, count: Optional[int], followers_limit: int) -> Set[str]:
    tweets = extract_tweets(tweets_dir, count)
    retweets_map = get_retweet_ids(tweets)

    return get_authors_to_process(
        tweets, retweets_map, max_followers=followers_limit
    )

def main(args):
    authors_to_process: Set[str] = reduce(set.__or__, [
        get_topic_authors(dir_, args.count, args.followers_limit)
        for dir_ in args.tweets_dirs
    ])

    print(f"processing {len(authors_to_process)} authors")

    # Bottleneck of the script
    fetch_followers_of_authors(
        authors_to_process,
        args.followers_limit,
        args.retries,
        Followers(args.followers_dir),
        args.force,
        threads=args.threads,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a Sysomos collection of tweets, fetch the "
        + "needed followers list, and build a graph representing it."
    )
    parser.add_argument("--tweets-dirs", "--tweets", "-i", type=str, required=True, nargs="+")
    parser.add_argument("--followers_dir", "-o", type=str, required=True)
    parser.add_argument("--count", "-c", nargs="?", type=int, default=None)
    parser.add_argument("--min-cascade-length", "--len", nargs="?", type=int, default=0)
    parser.add_argument(
        "--followers-limit", "--limit", nargs="?", type=int, default=5000
    )
    parser.add_argument("--force-update", "-f", action="store_true", dest="force")
    parser.add_argument("--threads", nargs="?", type=int, default=None)
    parser.add_argument("--retries", nargs="?", type=int, default=1)

    main(parser.parse_args())
