import argparse
from spread_classification.utils import extract_tweet_ids
from uuid import uuid4
import csv
from pathlib import Path


def main(args):
    tweet_ids = extract_tweet_ids(args.sysomos_file, should_sort=True, count=args.count)
    writer = csv.writer(args.output_file)
    writer.writerow([uuid4(), "", Path(args.sysomos_file.name).stem, "\t".join(tweet_ids)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sysomos-file", "-i", type=argparse.FileType("r"))
    parser.add_argument("--output-file", "-o", type=argparse.FileType("a"))
    parser.add_argument("--count", "-c", type=int, default=None)

    main(parser.parse_args())
