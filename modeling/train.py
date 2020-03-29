import argparse
import logging
import os
import random
from datetime import datetime
import itertools as it

import torch
from torch import nn

from src.configuration import Configuration
from src.ignite_training import setup_logger
from src.training_utils import get_data_loaders, train_on_loaders


def main(args):
    now = datetime.now()
    for size in args.sizes:
        log_dir = os.path.join(
            args.output,
            "training_{}_{}_{}".format(
                args.model_name, size, now.strftime("%Y%m%d_%H%M")
            ),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_level = logging.INFO
        if args.debug:
            log_level = logging.DEBUG
            print("Activated debug mode")

        logger = logging.getLogger("Spread Classification: Train")
        setup_logger(logger, log_dir, log_level)

        device = "cpu"
        if torch.cuda.is_available():
            logger.debug("CUDA is enabled")
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = "cuda"

        logger.debug("Setup model: {}".format(args.model_name))

        logger.debug("Setup train/val dataloaders")
        loaders = get_data_loaders(
            args.dataset_path,
            args.train_batch_size,
            args.val_batch_size,
            args.num_workers,
            dataset_size=size,
            device=device,
            n_splits=5,
        )

        logger.debug("Setup criterion")
        criterion = nn.BCELoss()
        if "cuda" in device:
            criterion = criterion.cuda()

        for run_index, (train_loader, val_loader) in it.islice(enumerate(loaders), 0, args.max_runs):
            configuration = Configuration.from_dict(
                **{
                    **vars(args),
                    "run_index": run_index,
                    "log_dir": log_dir,
                    "log_level": log_level,
                    "device": device,
                    "dataset_size": size,
                }
            )

            train_on_loaders(train_loader, val_loader, criterion, logger, configuration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-i", type=str, required=True)
    parser.add_argument("--epochs", "-e", type=int, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument(
        "--optimizer", type=str, required=True, help='Any of "adam" or "sgd"'
    )
    parser.add_argument("--sizes", "-s", type=int, nargs="*", default=[None])
    parser.add_argument("--max-runs", type=int, default=None)

    parser.add_argument("--model-name", "-m", type=str, default="SpreadClassification")
    parser.add_argument("--train-batch-size", "-t", type=int, default=8)
    parser.add_argument("--val-batch-size", "-v", type=int, default=4)
    parser.add_argument("--num-workers", "-w", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-update-every", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma", "-g", type=float, default=0.9)
    parser.add_argument("--restart-every", type=int, default=10)
    parser.add_argument("--restart-factor", type=float, default=1.5)
    parser.add_argument("--init-lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-reduce-patience", type=int, default=10)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--debug", action="store_true", dest="debug", default=False)

    main(parser.parse_args())
