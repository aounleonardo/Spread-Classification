import logging
import os

import dgl
import numpy as np
import torch
import visdom
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer
from ignite.contrib.metrics import ROC_AUC
from ignite.metrics import Accuracy, Loss, Precision, Recall
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch import nn, tensor

from .lr_scheduler import LRSchedulerWithRestart
from .nlp_ignite_engines import create_nlp_trainer, create_nlp_evaluator

TRAINER_FACTORY_MAP = {
    "graph": lambda model, optimizer, loss_fn, device: create_supervised_trainer(
        model, optimizer, loss_fn, device=device, prepare_batch=_prepare_batch,
    ),
    "text": create_nlp_trainer,
}
EVALUATOR_FACTORY_MAP = {
    "graph": lambda model, metrics, device: create_supervised_evaluator(
        model, metrics, device=device, prepare_batch=_prepare_batch,
    ),
    "text": create_nlp_evaluator,
}


def _prepare_batch(batch, device, non_blocking):
    return dgl.unbatch(batch[0]), batch[1].float().to(device)


def _output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


def _negative_output_transform(output):
    return tuple(1 - labels for labels in _output_transform(output))


def get_criterion(configuration):
    ratio = configuration.positive_ratio
    device = configuration.device

    def get_loss(predictions, truth):
        weights = tensor([ratio + gt * (1 - 2 * ratio) for gt in truth])
        criterion = nn.BCELoss(weight=weights)
        if "cuda" in device:
            criterion.cuda()
        return criterion(predictions, truth.float())

    return get_loss


def get_trainer(model, optimizer, configuration):
    assert configuration.data_type in TRAINER_FACTORY_MAP, "Data type not in {}".format(
        TRAINER_FACTORY_MAP.keys()
    )
    return TRAINER_FACTORY_MAP[configuration.data_type](
        model, optimizer, get_criterion(configuration), device=configuration.device,
    )


def get_evaluators(model, configuration):
    assert (
        configuration.data_type in EVALUATOR_FACTORY_MAP
    ), "Data type not in {}".format(EVALUATOR_FACTORY_MAP.keys())
    metrics = {
        "accuracy": Accuracy(_output_transform),
        "precision": Precision(_output_transform),
        "recall": Recall(_output_transform),
        "loss": Loss(get_criterion(configuration)),
        "auc": ROC_AUC(),
        "tnr": Recall(_negative_output_transform),
        "npv": Precision(_negative_output_transform),
    }
    train_evaluator = EVALUATOR_FACTORY_MAP[configuration.data_type](
        model, metrics=metrics, device=configuration.device,
    )
    val_evaluator = EVALUATOR_FACTORY_MAP[configuration.data_type](
        model, metrics=metrics, device=configuration.device,
    )

    return train_evaluator, val_evaluator


def add_lr_scheduling(trainer, val_evaluator, optimizer, configuration):
    lr_scheduler = ExponentialLR(optimizer, gamma=configuration.gamma)

    if configuration.lr_update_every > 0:
        lr_scheduler_restarts = LRSchedulerWithRestart(
            lr_scheduler,
            restart_every=configuration.restart_every,
            restart_factor=configuration.restart_factor,
            init_lr_factor=configuration.init_lr_factor,
        )

        @trainer.on(Events.EPOCH_STARTED)
        def update_lr_schedulers(engine):
            if (engine.state.epoch - 1) % configuration.lr_update_every == 0:
                lr_scheduler_restarts.step()

    if configuration.lr_reduce_patience > 0:
        reduce_on_plateau = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=configuration.lr_reduce_factor,
            patience=configuration.lr_reduce_patience,
            threshold=0.01,
            verbose=True,
        )

        @val_evaluator.on(Events.COMPLETED)
        def update_reduce_on_plateau(engine):
            val_loss = engine.state.metrics["loss"]
            reduce_on_plateau.step(val_loss)


def _score_function(engine):
    val_loss = engine.state.metrics["loss"]
    # Objects with highest scores will be retained.
    return -val_loss


def add_early_stopping(trainer, val_evaluator, configuration):
    # Setup early stopping:
    handler = EarlyStopping(
        patience=configuration.early_stop_patience,
        score_function=_score_function,
        trainer=trainer,
    )
    setup_logger(handler._logger, configuration.log_dir, configuration.log_level)
    val_evaluator.add_event_handler(Events.COMPLETED, handler)


def add_model_checkpointing(trainer, val_evaluator, model, configuration):
    # Setup model checkpoint:
    best_model_saver = ModelCheckpoint(
        configuration.log_dir,
        filename_prefix=f"model_{configuration.run_index}",
        score_name="val_loss",
        score_function=_score_function,
        n_saved=5,
        atomic=True,
        create_dir=True,
        save_as_state_dict=True,
    )
    val_evaluator.add_event_handler(
        Events.COMPLETED, best_model_saver, {configuration.model_name: model}
    )

    last_model_saver = ModelCheckpoint(
        configuration.log_dir,
        filename_prefix=f"checkpoint_{configuration.run_index}",
        save_interval=1,
        n_saved=1,
        atomic=True,
        create_dir=True,
        save_as_state_dict=True,
    )
    trainer.add_event_handler(
        Events.COMPLETED, last_model_saver, {configuration.model_name: model}
    )


def setup_logger(logger, output, level=logging.INFO):
    logger.setLevel(level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(output, "train.log"))
    fh.setLevel(level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s|%(name)s|%(levelname)s| %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
