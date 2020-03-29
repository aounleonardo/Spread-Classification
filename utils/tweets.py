import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional, Callable


def get_datetime(twitter_obj: Dict[str, Any]) -> dt.datetime:
    twitter_format = "%a %b %d %H:%M:%S %z %Y"
    return dt.datetime.strptime(twitter_obj["created_at"], twitter_format)


def _get_offspring_tweets(tweet) -> List[Dict[str, Any]]:
    offsprings = [tweet.get("retweeted_status"), tweet.get("quoted_status")]
    offsprings = [off for off in offsprings if off is not None]
    recursive = [
        rec
        for off in offsprings
        for rec in _get_offspring_tweets(off)
        if rec is not None
    ]
    return offsprings + recursive


def _extract_from_filename(tweet_filename: str) -> List[Dict[str, Any]]:
    with open(tweet_filename) as file:
        tweet = json.load(file)
    return [tweet] + _get_offspring_tweets(tweet)


def extract_tweets(
    tweets_dir: str, count: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    filenames = [
        f"{tweets_dir}/{name}"
        for name in os.listdir(tweets_dir)
        if name.endswith(".json")
    ]
    tweets = (
        tweet
        for filename in filenames
        for tweet in _extract_from_filename(filename)
        if tweet is not None
    )
    if count is not None:
        tweets = sorted(tweets, key=get_datetime)[:count]

    return {tweet["id_str"]: tweet for tweet in tweets}


def get_retweeted_id(tweet: Dict[str, Any]) -> Optional[str]:
    retweeted = tweet.get("retweeted_status")
    if retweeted is None:
        return None
    return retweeted.get("id_str")


def get_retweet_ids(tweets: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    return _group_by_tweet(tweets, get_retweeted_id)


def get_children_ids(tweets: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    return _group_by_tweet(tweets, get_parent_tweet_id)


def _group_by_tweet(
    tweets: Dict[str, Dict[str, Any]], select: Callable[[Dict[str, Any]], Optional[str]]
) -> Dict[str, List[str]]:
    sorted_tweets = sorted(tweets.values(), key=get_datetime)
    groups = {tweet_id: [] for tweet_id in tweets.keys()}

    for tweet in sorted_tweets:
        tweet_id = tweet["id_str"]
        groups[tweet_id] = []
        leader_id = select(tweet)

        if leader_id is not None:
            if leader_id not in groups:
                groups[leader_id] = []
            groups[leader_id].append(tweet_id)
    return groups


def get_parent_tweet_id(tweet: Dict[str, Any]) -> Optional[str]:
    retweeted = tweet.get("retweeted_status")
    if retweeted:
        return retweeted["id_str"]

    quoted = tweet.get("quoted_status")
    if quoted:
        return quoted["id_str"]

    replied_id = tweet.get("in_reply_to_status_id_str")
    if replied_id:
        return replied_id

    return None

