import csv
import datetime as dt
from typing import List, IO, OrderedDict, Optional


headers = [
    "No",
    "Source",
    "Host",
    "Link",
    "Date(ET)",
    "Time(ET)",
    "LocalTime",
    "Category",
    "Author ID",
    "Author Name",
    "Author URL",
    "Authority",
    "Followers",
    "Following",
    "Age",
    "Gender",
    "Language",
    "Country",
    "Province/State",
    "City",
    "Location",
    "Sentiment",
    "Themes",
    "Classifications",
    "Entities",
    "Alexa Rank",
    "Alexa Reach",
    "Title",
    "Snippet",
    "Contents",
    "Summary",
    "Bio",
    "Unique ID",
    "Post Source",
]


def extract_tweet_ids(sysomos_file: IO[str], should_sort: bool, count: Optional[int]) -> List[str]:
    while True:
        line = sysomos_file.readline()
        if all(h in line for h in headers):
            break
    reader = csv.DictReader(sysomos_file, fieldnames=headers)
    tweets = (tweet for tweet in reader)
    if should_sort:
        tweets = sorted(tweets, key=tweet_to_datetime)
    return [tweet["Unique ID"] for tweet in tweets][:count]


def tweet_to_datetime(tweet: OrderedDict[str, str]) -> dt.datetime:
    return dt.datetime.fromisoformat(f'{tweet["Date(ET)"]} {tweet["Time(ET)"]}')
