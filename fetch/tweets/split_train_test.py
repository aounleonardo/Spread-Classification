import argparse
import os
from spread_classification.utils import UrlsDataset
from torch.utils.data import Subset
import shutil

def get_train_and_test(dataset, train_ratio):
    return dataset.get_subsets([len(dataset) * .8, len(dataset) * .2], None)

def copy_instance(dataset: UrlsDataset, key: str, output_dir: str):
    label = dataset.labels[key]
    dir_ = os.path.join(output_dir, dataset.labels_map[label], key)
    if not os.path.exists(dir_):
        os.makedirs(dir_)
        shutil.copytree(dataset[key], os.path.join(dir_, "tweets"))


def copy_data(subset: Subset, destination: str):
    if not os.path.exists(destination):
        os.mkdir(destination)
    for key in subset.indices:
        copy_instance(subset.dataset, key, destination)

def main(args):
    dataset = UrlsDataset(args.dataset)
    train, test = get_train_and_test(dataset, args.train_ratio)

    if not os.path.exists(args.output):
        os.mkdir(args.output)
    copy_data(train, os.path.join(args.output, "train"))
    copy_data(test, os.path.join(args.output, "test"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-i", type=str, required=True)
    parser.add_argument("--train-ratio", "-r", type=float, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    main(parser.parse_args())