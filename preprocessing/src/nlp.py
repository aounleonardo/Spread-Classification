from typing import Dict, Any


def get_tweet_content(tweet: Dict[str, Any]) -> str:
    return tweet.get("text", "")


def get_user_bio(tweet: Dict[str, Any]) -> str:
    return tweet.get("user", {"description": ""}).get("description", "")


def get_bio_and_content(tweet: Dict[str, Any]) -> str:
    return tweet["user"]["description"] + "\n" + tweet["text"]

