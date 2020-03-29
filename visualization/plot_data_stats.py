import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# plt.style.use("dark_background")


def load_stats(stats_dir, run_name):
    with open(os.path.join(stats_dir, f"{run_name}.json")) as file:
        return json.load(file)


def non_zero(arr):
    return arr[arr != 0]


def plot_histograms(arrays, keys, stats_dir, run_name, fig_name):
    rows = 1 + (len(arrays) // 2)
    f, axes = plt.subplots(rows, 2)
    f.set_figwidth(24)
    f.set_figheight(5 * rows)
    reshaped_axes = axes.flatten()
    for index, array in enumerate(arrays):
        reshaped_axes[index].set_title(keys[index])
        n, bins, patches = reshaped_axes[index].hist(array, bins=100, log=True)
    for ax in reshaped_axes[len(arrays) :]:
        f.delaxes(ax)
    plt.savefig(os.path.join(stats_dir, f"{run_name}_{fig_name}.png"))


def plot(array, keys, title, stats_dir, run_name, fig_name):
    f = plt.figure(figsize=(12, 12))
    f.suptitle(title)
    plt.plot(keys, array)
    plt.savefig(os.path.join(stats_dir, f"{run_name}_{fig_name}.png"))


def process_sizes(stats, prune_lengths, stats_dir, run_name):
    branches_keys = ["cascades", "spreads"]
    tree_keys = [
        "full_size",
        *[f"pruned_{length}_size" for length in prune_lengths],
    ]
    sizes_keys = [
        *branches_keys,
        *tree_keys,
    ]

    sizes_arrays = np.array(
        [
            *[np.array(list(stats[key].values())) for key in branches_keys],
            *[non_zero(np.array(stats[key])) for key in tree_keys],
        ]
    )
    plot_histograms(sizes_arrays, sizes_keys, stats_dir, run_name, "sizes")

    pruned_lens = [sum(stats[key]) for key in tree_keys]
    plot(
        pruned_lens,
        [str(x) for x in [1] + list(prune_lengths)],
        "total tweets per prune level",
        stats_dir,
        run_name,
        "count",
    )


def process_followers(stats, min_followers, max_followers, stats_dir, run_name):
    needed_followers = np.array(stats["needed_followers"])
    available_followers = np.array(stats["available_followers"])
    followers_indices = np.where(needed_followers >= min_followers)
    needed_followers = needed_followers[followers_indices]
    available_followers = available_followers[followers_indices]

    less_than_max_followers = needed_followers[
        np.where(needed_followers <= max_followers)
    ]
    followers_ratio = available_followers / needed_followers

    followers_keys = np.array(
        [
            "gathered_followers_ratio",
            f"less_than_{min_followers}_followers",
            f"less_than_{max_followers}_followers",
        ]
    )
    followers_arrays = np.array(
        [followers_ratio, needed_followers, less_than_max_followers]
    )
    plot_histograms(followers_arrays, followers_keys, stats_dir, run_name, "followers")


def process_stats(stats, max_prune_length, stats_dir, run_name):

    prune_lengths = range(2, max_prune_length + 1)
    process_sizes(stats, prune_lengths, stats_dir, run_name)

    process_followers(stats, 5, 105, stats_dir, run_name)


def main(args):
    run_names = args.run_names or ["train, test"]
    for run_name in run_names:
        stats = load_stats(args.stats_dir, run_name)
        process_stats(stats, args.max_prune_length, args.stats_dir, run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", "-i", type=str, required=True)
    parser.add_argument("--max-prune-length", "--len", type=int, required=True)
    parser.add_argument("--run-names", "-n", type=str, nargs="*", required=False)
    main(parser.parse_args())
