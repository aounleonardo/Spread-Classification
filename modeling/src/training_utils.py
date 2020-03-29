import os

import numpy as np
import torch
from ignite._utils import convert_tensor
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from spread_classification.preprocessing import FEATURES_LIST
from spread_classification.utils import GraphDataset, collate

from .classifier import Classifier
from .ignite_training import (
    add_early_stopping,
    add_lr_scheduling,
    add_model_checkpointing,
    get_evaluators,
    get_trainer,
)
from .monitor import Monitor

OPTIMIZER_MAP = {"adam": Adam, "sgd": SGD}

RATINGS_MAP = {"False": 0, "True": 1}


def get_data_loaders(
    dataset_path,
    train_batch_size,
    val_batch_size,
    num_workers,
    dataset_size=None,
    n_splits=5,
    device=None,
):
    dataset = GraphDataset(dataset_path)
    dataset = dataset.get_subsets([dataset_size or len(dataset)], preserve_ratios=False)[0]

    generic_loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
    )
    indices = get_train_val_indices(generic_loader, n_splits)
    pin_memory = "cuda" in device if device is not None else False

    return [
        (
            DataLoader(
                dataset,
                batch_size=train_batch_size,
                sampler=SubsetRandomSampler(train_indices),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate,
            ),
            DataLoader(
                dataset,
                batch_size=val_batch_size,
                sampler=SubsetRandomSampler(val_indices),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate,
            ),
        )
        for (train_indices, val_indices) in indices
    ]


def get_train_val_indices(data_loader, n_splits):
    # Stratified split: train/val:
    n_samples = len(data_loader.dataset)
    batch_size = data_loader.batch_size
    X = np.zeros((n_samples, 1))
    y = np.zeros(n_samples, dtype=np.int)
    for i, (_, label) in enumerate(data_loader):
        start_index = batch_size * i
        end_index = batch_size * (i + 1)
        y[start_index:end_index] = label.numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    return [
        (train_indices, val_indices) for (train_indices, val_indices) in skf.split(X, y)
    ]


def train_on_loaders(train_loader, val_loader, criterion, logger, configuration):
    model = Classifier()
    if "cuda" in configuration.device:
        model = model.to(configuration.device)

    logger.debug("Setup tensorboard writer")
    writer = SummaryWriter(
        log_dir=os.path.join(
            configuration.log_dir, f"tensorboard_{configuration.run_index}"
        )
    )

    write_model_graph(writer, model, train_loader, device=configuration.device)

    logger.debug("Setup optimizer")
    assert configuration.optimizer in OPTIMIZER_MAP, "Optimizer name not in {}".format(
        OPTIMIZER_MAP.keys()
    )
    optimizer = OPTIMIZER_MAP[configuration.optimizer](
        model.parameters(), lr=configuration.lr, weight_decay=configuration.weight_decay
    )

    logger.debug("Setup ignite trainer and evaluator")

    trainer = get_trainer(model, optimizer, criterion, configuration.device)
    train_evaluator, val_evaluator = get_evaluators(
        model, criterion, configuration.device
    )

    monitor = Monitor(
        model,
        trainer,
        optimizer,
        train_evaluator,
        train_loader,
        val_evaluator,
        val_loader,
        configuration,
    )
    monitor.add_logging(logger)
    monitor.add_visualizing()
    monitor.add_csv_writing()
    monitor.add_tensorboard_writing()

    logger.debug("Setup handlers")

    add_lr_scheduling(trainer, val_evaluator, optimizer, configuration)

    if configuration.early_stop_patience > 0:
        add_early_stopping(trainer, val_evaluator, configuration)

    add_model_checkpointing(trainer, val_evaluator, model, configuration)

    logger.info("Start training: {} epochs".format(configuration.epochs))
    try:
        trainer.run(train_loader, max_epochs=configuration.epochs)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception(e)
        if configuration.debug:
            try:
                # open an ipython shell if possible
                import IPython

                IPython.embed()  # noqa
            except ImportError:
                print("Failed to start IPython console")

    logger.debug("Training is ended")


def write_model_graph(writer, model, data_loader, device):
    data_loader_iter = iter(data_loader)
    x, _ = next(data_loader_iter)
    x = torch.zeros(
        1
    )  # This is the only way it works for now :/ convert_tensor cannot take in a graph
    x = convert_tensor(x, device=device)
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print("Failed to save model graph: {}".format(e))
