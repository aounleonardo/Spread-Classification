import argparse
import os
from torch.utils.data import Subset
from spread_classification.utils import UrlsDataset
import shutil


def trim_dataset(dataset: UrlsDataset, size_ratio: float) -> Subset:
    subset_size = int(size_ratio * len(dataset))
    return dataset.get_subsets([subset_size], label_ratios=None)[0]


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
    for dataset_name in ["train", "test"]:
        dataset = UrlsDataset(os.path.join(args.dataset_path, dataset_name))
        subset = trim_dataset(dataset, args.ratio)

        if not os.path.exists(args.output):
            os.mkdir(args.output)
        copy_data(subset, os.path.join(args.output, dataset_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--ratio", "-r", type=float, required=True)
    main(parser.parse_args())
