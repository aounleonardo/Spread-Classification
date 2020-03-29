import argparse
import os
from spread_classification.utils import Followers
import shutil
import random

def transfer(from_dataset, to_dataset, ratio):
    all_keys = from_dataset.keys()
    sampled_keys = random.sample(all_keys, int(ratio * len(all_keys)))
    for key in sampled_keys:
        to_dataset[key] = from_dataset[key]

def main(args):
    read_dataset = Followers(args.dataset)
    write_dataset = Followers(args.output)

    transfer(read_dataset, write_dataset, args.ratio)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-i", type=str, required=True)
    parser.add_argument("--ratio", "-r", type=float, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    main(parser.parse_args())