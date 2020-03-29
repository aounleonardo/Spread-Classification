from pathlib import Path
import argparse
import csv
from src.observers import VisdomVisualiser


def visualise_line(vis, tag, value, epoch):
    if "learning_rate" in tag:
        if "group" in tag:
            group_nb = tag.split("_")[-1]
            vis.lr_group(value, group_nb, epoch)
        else:
            vis.lr(value, epoch)
    else:
        mode, metric = tag.split("/")
        if "training" == mode:
            vis.training_metric(metric, value, epoch)
        elif "validation" == mode:
            vis.validation_result(metric, value, epoch)


def main(args):
    headers = ["tag", "value", "stamp"]
    env_name = Path(args.csv_file.name).stem
    reader = csv.DictReader(args.csv_file, fieldnames=headers)
    vis = VisdomVisualiser(env_name, args.lr_groups_nb)
    for line in reader:
        visualise_line(vis, line["tag"], float(line["value"]), int(line["stamp"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", "-i", required=True, type=argparse.FileType("r"))
    parser.add_argument("--lr-groups-nb", default=1, type=int)

    main(parser.parse_args())
