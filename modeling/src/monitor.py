import csv
import logging
import os

import numpy as np
import torch
import visdom
from ignite._utils import convert_tensor
from ignite.engine import Events
from ignite.handlers import Timer
from tensorboardX import SummaryWriter

from .observers import CsvWriter, Logger, TensorboardWriter, VisdomVisualiser


class Monitor:
    def __init__(
        self,
        model,
        trainer,
        optimizer,
        train_evaluator,
        train_loader,
        val_evaluator,
        val_loader,
        configuration,
    ):
        self._trainer = trainer
        self._model = model
        self._optimizer = optimizer
        self._train_evaluator = train_evaluator
        self._train_loader = train_loader
        self._val_evaluator = val_evaluator
        self._val_loader = val_loader
        self._env_name = f"{configuration.model_name}_{configuration.dataset_size}_{configuration.run_index}"
        self._configuration = configuration

        self._timer = self._start_timer()
        self._logger = None
        self._tensorboard_writer = None
        self._csv_writer = None
        self._visdom_visualiser = None

    def _add_epoch_observer(self, observer):
        @self._trainer.on(Events.EPOCH_STARTED)
        def lrs(engine):
            epoch = engine.state.epoch
            if len(self._optimizer.param_groups) == 1:
                lr = float(self._optimizer.param_groups[0]["lr"])
                observer.lr(lr, epoch)
            else:
                for i, param_group in enumerate(self._optimizer.param_groups):
                    lr = float(param_group["lr"])
                    observer.lr_group(lr, i, epoch)

        @self._trainer.on(Events.EPOCH_COMPLETED)
        def training_metrics(engine):
            epoch = engine.state.epoch
            metrics = Monitor._get_metrics(self._train_evaluator, self._train_loader)
            observer.training_metrics(metrics, epoch)

        @self._trainer.on(Events.EPOCH_COMPLETED)
        def validation_results(engine):
            epoch = engine.state.epoch
            metrics = Monitor._get_metrics(self._val_evaluator, self._val_loader)
            observer.validation_results(metrics, epoch)

        return self

    def add_logging(self, logger):
        self._logger = Logger(logger, self._timer, self._configuration)

        @self._trainer.on(Events.ITERATION_COMPLETED)
        def training_loss(engine):
            iter_nb = (engine.state.iteration - 1) % len(self._train_loader) + 1
            if iter_nb % self._configuration.log_interval == 0:
                self._logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.4f}".format(
                        engine.state.epoch,
                        iter_nb,
                        len(self._train_loader),
                        engine.state.output,
                    )
                )

        return self._add_epoch_observer(self._logger)

    def add_visualizing(self):
        self._visdom_visualiser = VisdomVisualiser(
            self._env_name, len(self._optimizer.param_groups)
        )
        return self._add_epoch_observer(self._visdom_visualiser)

    def add_tensorboard_writing(self):
        self._tensorboard_writer = TensorboardWriter(
            self._env_name, self._configuration
        )
        self._write_model_graph()
        return self._add_epoch_observer(self._tensorboard_writer)

    def add_csv_writing(self):
        self._csv_writer = CsvWriter(self._configuration.log_dir, self._env_name)
        return self._add_epoch_observer(self._csv_writer)

    @staticmethod
    def _get_metrics(evaluator, loader):
        metrics = evaluator.run(loader).metrics
        return {
            name: metrics[name]
            for name in ["loss", "accuracy", "precision", "recall", "auc", "tnr", "npv"]
        }

    def _start_timer(self):
        timer = Timer(average=True)
        timer.attach(
            self._trainer,
            start=Events.EPOCH_STARTED,
            resume=Events.ITERATION_STARTED,
            pause=Events.ITERATION_COMPLETED,
        )
        return timer

    def _write_model_graph(self):
        data = next(iter(self._train_loader))
        x = data[0]
        x = torch.zeros(
            1
        )  # This is the only way it works for now :/ convert_tensor cannot take in a graph
        x = convert_tensor(x, device=self._configuration.device)
        try:
            self._tensorboard_writer.add_graph(self._model, x)
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
