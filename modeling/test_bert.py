import argparse
import itertools as it
import logging
import os
import json
from datetime import datetime

import numpy as np
import torch

from spread_classification.utils import StoriesDataset
from src.configuration import Configuration
from src.ignite_training import setup_logger
from src.nlp import TextClassifier
from src.training_utils import get_text_test_loader, test_on_loader, get_model_cls


def flat_accuracy(preds, labels):
    pred_flat = np.round(preds).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def main(args):
    now = datetime.now()

    for size in args.sizes:
        device = "cpu"
        if torch.cuda.is_available():
            from torch.backends import cudnn

            cudnn.benchmark = True
            device = "cuda"

        model = get_model_cls(args.model_class).from_file(args.model_path.name, device, pretrained=not "_untrained" in args.model_class)
        # model = torch.load(args.model_path.name, map_location=device)

        positive_ratio, test_loader = get_text_test_loader(
            args.dataset_path,
            StoriesDataset,
            args.batch_size,
            args.num_workers,
            model.MAX_LEN,
            dataset_size=size,
            positive_ratio=args.positive_ratio,
            collate_fn=None,
            pin_memory=False,
            model_class=args.model_class,
        )
        dataset_size = size or len(test_loader.dataset)

        log_dir = os.path.join(
            args.output,
            "testing_{}_{}_{}".format(
                args.model_name, dataset_size, now.strftime("%Y%m%d_%H%M")
            ),
        )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_level = logging.INFO
        if args.debug:
            log_level = logging.DEBUG
            print("Activated debug mode")

        logger = logging.getLogger("Spread Classification: Test")
        setup_logger(logger, log_dir, log_level)

        logger.debug("Setup model: {}".format(args.model_name))

        configuration = Configuration.from_dict(
            **{
                **vars(args),
                "optimizer": "N/A",
                "epochs": -1,
                "train_batch_size": -1,
                "val_batch_size": args.batch_size,
                "lr": -1,
                "lr_update_every": -1,
                "weight_decay": -1,
                "gamma": -1,
                "restart_every": -1,
                "restart_factor": -1,
                "init_lr_factor": -1,
                "lr_reduce_patience": -1,
                "lr_reduce_factor": -1,
                "early_stop_patience": -1,
                "positive_ratio": positive_ratio,
                "run_index": -1,
                "log_dir": log_dir,
                "log_level": log_level,
                "log_interval": -1,
                "device": device,
                "dataset_size": dataset_size,
                "data_type": "text",
                "model_class": "text",
                "model_name": args.model_path.name,
                "gcn_hidden_dimension": -1,
                "fc_hidden_dimension": -1,
            }
        )

        metrics = test_on_loader(model, test_loader, logger, configuration)
        with open(os.path.join(log_dir, "metrics.json"), "w") as file:
            json.dump(metrics, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", "--model", type=argparse.FileType("rb"), required=True
    )
    parser.add_argument("--dataset-path", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--sizes", "-s", type=int, nargs="*", default=[None])
    parser.add_argument("--positive-ratio", "-r", type=float, default=None)

    parser.add_argument("--model-name", "-m", type=str, default="SpreadClassification")
    parser.add_argument("--batch-size", "-b", type=int, default=32)
    parser.add_argument("--num-workers", "-w", type=int, default=1)
    parser.add_argument("--debug", action="store_true", dest="debug", default=False)
    parser.add_argument("--model-class", "-c", type=str, default="text")

    main(parser.parse_args())
