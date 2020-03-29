import os

import numpy as np
import torch
from ignite._utils import convert_tensor
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from torch.optim import SGD, Adam, SparseAdam
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW
from typing import Optional

from spread_classification.preprocessing import FEATURES_LIST
from spread_classification.utils import GraphsDataset, collate, StoriesDataset

from .nlp import get_attention_masks, TextClassifier
from .fasttext import FastTextClassifier
from .classifier import Classifier, PureContextClassifier
from .ignite_training import (
    add_early_stopping,
    add_lr_scheduling,
    add_model_checkpointing,
    get_evaluators,
    get_trainer,
)
from .monitor import Monitor

OPTIMIZER_MAP = {"adam": Adam, "adamw": AdamW, "sgd": SGD, "sparsadam": SparseAdam}

RATINGS_MAP = {"False": 0, "True": 1}


def _untrained(cls, **kwargs):
    def aux(**kwargs):
        return cls(pretrained=False, **kwargs)
    return aux


MODEL_MAP = {
    "graph": Classifier,
    "graph_without_text": PureContextClassifier,
    "text": TextClassifier,
    "text_untrained": _untrained(TextClassifier),
    "fasttext": FastTextClassifier,
    "fasttext_untrained": _untrained(FastTextClassifier),
}

def _get_model_cls(model_class):
    assert model_class in MODEL_MAP, "Model Class not in {}".format(MODEL_MAP.keys())

    return MODEL_MAP[model_class]

def get_model_cls(model_class):
    return _get_model_cls(model_class.replace("_untrained", ""))

def _get_binary_ratio_and_subset(
    dataset, dataset_size: Optional[int], positive_ratio: Optional[float]
):
    label_ratios = (
        None if positive_ratio is None else [1 - positive_ratio, positive_ratio]
    )
    if label_ratios is None:
        label_ratios = list(dataset.get_label_ratios().values())
    subset = dataset.get_subsets(
        [dataset_size or len(dataset)], label_ratios=label_ratios
    )[0]
    return label_ratios[1], subset


def get_graph_loaders(
    dataset_path,
    train_batch_size,
    val_batch_size,
    num_workers,
    dataset_size=None,
    positive_ratio=None,
    n_splits=5,
    device=None,
):
    dataset = GraphsDataset(dataset_path)
    corrected_ratio, subset = _get_binary_ratio_and_subset(
        dataset, dataset_size, positive_ratio
    )

    pin_memory = "cuda" in device if device is not None else False
    return (
        corrected_ratio,
        _get_data_loaders(
            subset,
            train_batch_size,
            val_batch_size,
            num_workers,
            n_splits=n_splits,
            collate_fn=collate,
            pin_memory=pin_memory,
        ),
    )


def _convert_text_to_tensor_dataset(subset, convert_to_ids, max_text_len):
    sequences, labels = zip(*subset)
    input_ids = convert_to_ids(sequences, max_text_len)
    attention_masks = get_attention_masks(input_ids)

    return TensorDataset(
        torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)
    )


def get_text_loaders(
    dataset_path,
    train_batch_size,
    val_batch_size,
    num_workers,
    max_text_len,
    dataset_size=None,
    positive_ratio=None,
    n_splits=5,
    device=None,
    model_class="text",
):
    dataset = StoriesDataset(dataset_path)
    corrected_ratio, subset = _get_binary_ratio_and_subset(
        dataset, dataset_size, positive_ratio
    )
    convert_ids_fn = get_model_cls(model_class).get_convert_ids_fn()
    tensor_dataset = _convert_text_to_tensor_dataset(
        subset, convert_ids_fn, max_text_len
    )

    return (
        corrected_ratio,
        _get_data_loaders(
            tensor_dataset,
            train_batch_size,
            val_batch_size,
            num_workers,
            n_splits=n_splits,
        ),
    )


def get_graph_test_loader(
    dataset_path,
    dataset_class,
    test_batch_size,
    num_workers,
    dataset_size=None,
    positive_ratio=None,
    collate_fn=None,
    pin_memory=False,
):
    dataset = dataset_class(dataset_path)
    positive_ratio, subset = _get_binary_ratio_and_subset(
        dataset, dataset_size, positive_ratio
    )

    return (
        positive_ratio,
        DataLoader(
            subset,
            batch_size=test_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
    )


def get_text_test_loader(
    dataset_path,
    dataset_class,
    test_batch_size,
    num_workers,
    max_text_len,
    dataset_size=None,
    positive_ratio=None,
    collate_fn=None,
    pin_memory=False,
    model_class="text",
):
    dataset = dataset_class(dataset_path)
    positive_ratio, subset = _get_binary_ratio_and_subset(
        dataset, dataset_size, positive_ratio
    )
    convert_ids_fn = get_model_cls(model_class).get_convert_ids_fn()
    tensor_dataset = _convert_text_to_tensor_dataset(
        subset, convert_ids_fn, max_text_len
    )

    return (
        positive_ratio,
        DataLoader(
            tensor_dataset,
            batch_size=test_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        ),
    )


def _get_data_loaders(
    dataset,
    train_batch_size,
    val_batch_size,
    num_workers,
    n_splits=5,
    collate_fn=None,
    pin_memory=False,
):
    generic_loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    indices = get_train_val_indices(generic_loader, n_splits)

    return [
        (
            DataLoader(
                dataset,
                batch_size=train_batch_size,
                sampler=SubsetRandomSampler(train_indices),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            ),
            DataLoader(
                dataset,
                batch_size=val_batch_size,
                sampler=SubsetRandomSampler(val_indices),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
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
    for i, data in enumerate(data_loader):
        label = data[-1]
        start_index = batch_size * i
        end_index = batch_size * (i + 1)
        y[start_index:end_index] = label.numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    return [
        (train_indices, val_indices) for (train_indices, val_indices) in skf.split(X, y)
    ]


def train_on_loaders(train_loader, val_loader, logger, configuration):
    model = _get_model_cls(configuration.model_class)(
        gcn_hidden_dimension=configuration.gcn_hidden_dimension,
        fc_hidden_dimension=configuration.fc_hidden_dimension,
    )
    if "cuda" in configuration.device:
        model = model.to(configuration.device)

    logger.debug("Setup optimizer")
    assert configuration.optimizer in OPTIMIZER_MAP, "Optimizer name not in {}".format(
        OPTIMIZER_MAP.keys()
    )
    optimizer = OPTIMIZER_MAP[configuration.optimizer](
        model.parameters(), lr=configuration.lr, weight_decay=configuration.weight_decay
    )

    logger.debug("Setup ignite trainer and evaluator")

    trainer = get_trainer(model, optimizer, configuration)
    train_evaluator, val_evaluator = get_evaluators(model, configuration)

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


def test_on_loader(model, test_loader, logger, configuration):
    if "cuda" in configuration.device:
        model = model.to(configuration.device)

    logger.debug("Setup ignite evaluator")
    evaluator = get_evaluators(model, configuration)[0]

    logger.info("Start testing")
    try:
        evaluator.run(test_loader)
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
    return evaluator.state.metrics
