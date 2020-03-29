from dataclasses import dataclass, fields
from operator import attrgetter
from spread_classification.utils import get_datetime
from enum import Enum
from typing import Optional, Dict, Any
from datetime import timedelta

TYPES_MAP = {"tweet": 1, "retweet": 0, "topic": -1}
DEVICES_MAP = {"iphone": 0, "android": 1, "web": 2, "ipad": 3, "bot": 4}


class NodeTypes(Enum):
    TWEET = 0
    REPLY = 1
    QUOTE = 2
    RETWEET = 3
    SHADOW = 4
    TOPIC = 5


class Devices(Enum):
    WEB_CLIENT = ("Twitter Web Client", 1)
    WEB_APP = ("Twitter Web App", 2)
    ANDROID = ("Twitter for Android", 3)
    IPHONE = ("Twitter for iPhone", 4)
    IPAD = ("Twitter for iPad", 5)
    BOT = ("bot", 6)
    OTHER = ("", 0)

    def __init__(self, search, code):
        self.search = search
        self.code = code


@dataclass
class Features:
    # Tweet features
    type_: int
    device: int = 0
    retweet_count: int = 0
    favorite_count: int = 0
    diff_time: float = 0.0
    # User features
    followers_count: int = 0
    friends_count: int = 0
    statuses_count: int = 0
    listed_count: int = 0
    favourites_count: int = 0
    twitter_age: float = 0.0
    verified: int = 0
    protected: int = 0


FEATURES_LIST = [feature.name for feature in fields(Features)]


def _get_type(tweet: Optional[Dict[str, Any]]) -> int:
    if tweet is None:
        type_ = NodeTypes.SHADOW
    elif tweet.get("type") == "root":
        type_ = NodeTypes.TOPIC
    elif tweet.get("retweeted_status") is not None:
        type_ = NodeTypes.RETWEET
    elif tweet.get("quoted_status") is not None:
        type_ = NodeTypes.QUOTE
    elif tweet.get("in_reply_to_status_id_str") is not None:
        type_ = NodeTypes.REPLY
    else:
        type_ = NodeTypes.TWEET

    return type_.value


def _get_device(tweet: Dict[str, Any]) -> int:
    type_ = Devices.OTHER
    if tweet is not None:
        post_source = tweet.get("source", "").lower()
        for device in Devices:
            if device.search.lower() in post_source:
                type_ = device
                break
    return type_.code


def _get_diff_time(
    tweet: Optional[Dict[str, Any]], parent: Optional[Dict[str, Any]]
) -> float:
    if (
        tweet is None
        or parent is None
        or tweet.get("type") == "root"
        or parent.get("type") == "root"
    ):
        return 0.0
    return (get_datetime(tweet) - get_datetime(parent)) / timedelta(hours=1)


def _get_user_twitter_age_at_tweet(
    user: Dict[str, Any], tweet: Dict[str, Any]
) -> float:
    return (get_datetime(tweet) - get_datetime(user)) / timedelta(days=365, hours=6)


def _is_retweet(tweet):
    return "RT @" in tweet.contents


def get_tweet_features(
    tweet: Optional[Dict[str, Any]], parent: Optional[Dict[str, Any]]
) -> Features:
    features = Features(_get_type(tweet))
    if tweet is None or tweet.get("id_str") is None:
        return features

    features.device = _get_device(tweet)
    features.retweet_count = tweet.get("retweet_count", 0)
    features.favorite_count = tweet.get("favorite_count", 0)
    features.diff_time = _get_diff_time(tweet, parent)
    features.verified = 1 if tweet.get("verified") else 0
    features.protected = 1 if tweet.get("protected") else 0

    user = tweet.get("user")
    if user is None:
        return features

    features.followers_count = user.get("followers_count", 0)
    features.friends_count = user.get("friends_count", 0)
    features.statuses_count = user.get("statuses_count", 0)
    features.listed_count = user.get("listed_count", 0)
    features.favourites_count = user.get("favourites_count", 0)
    features.twitter_age = _get_user_twitter_age_at_tweet(user, tweet)

    return features
