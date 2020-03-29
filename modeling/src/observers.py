import csv
import os

import numpy as np
from tensorboardX import SummaryWriter
from visdom import Visdom


class Observer:
    def lr(self, lr, epoch):
        raise NotImplementedError

    def lr_group(self, lr, group_nb, epoch):
        raise NotImplementedError

    def training_metrics(self, metrics, epoch):
        for name, value in metrics.items():
            self.training_metric(name, value, epoch)

    def validation_results(self, metrics, epoch):
        for name, value in metrics.items():
            self.validation_result(name, value, epoch)

    def training_metric(self, metric, value, epoch):
        raise NotImplementedError

    def validation_result(self, metric, value, epoch):
        raise NotImplementedError


class Logger(Observer):
    def __init__(self, logger, timer, description):
        self._logger = logger
        self._timer = timer

        self.info(description)

    def info(self, message):
        self._logger.info(message)

    def debug(self, message):
        self._logger.debug(message)

    def lr(self, lr, epoch):
        self._logger.debug("Learning rate: {}".format(lr))

    def lr_group(self, lr, group_nb, epoch):
        self._logger.debug("Learning rate (group {}): {}".format(group_nb, lr))

    def training_metrics(self, metrics, epoch):
        self._logger.info(
            "Epoch {} training time (seconds): {}".format(epoch, self._timer.value())
        )
        self._logger.info(
            "Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}".format(
                epoch, metrics["accuracy"], metrics["loss"]
            )
        )

    def validation_results(self, metrics, epoch):
        self._logger.info(
            "Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}".format(
                epoch, metrics["accuracy"], metrics["loss"]
            )
        )


class CsvWriter(Observer):
    def __init__(self, log_dir, env_name):
        self._csv_file = open(os.path.join(log_dir, f"{env_name}.csv"), "w")
        self._csv_writer = csv.writer(self._csv_file, dialect="unix")

    def __del__(self):
        self._csv_file.close()

    def lr(self, lr, epoch):
        self._csv_writer.writerow(["learning_rate", lr, epoch])

    def lr_group(self, lr, group_nb, epoch):
        self._csv_writer.writerow(
            ["learning_rate_group_{}".format(group_nb), lr, epoch]
        )

    def training_metric(self, metric, value, epoch):
        self._csv_writer.writerow([f"training/{metric}", value, epoch])

    def validation_result(self, metric, value, epoch):
        self._csv_writer.writerow([f"validation/{metric}", value, epoch])


class TensorboardWriter(Observer):
    def __init__(self, env_name, configuration):
        self._tensorboard_writer = SummaryWriter(
            log_dir=os.path.join(configuration.log_dir, f"tensorboard_{env_name}")
        )
        self._tensorboard_writer.add_text("Configuration", repr(configuration))

    def add_graph(self, model, input_example):
        self._tensorboard_writer.add_graph(model, input_example)

    def lr(self, lr, epoch):
        self._tensorboard_writer.add_scalar("learning_rate", lr, epoch)

    def lr_group(self, lr, group_nb, epoch):
        self._tensorboard_writer.add_scalar(
            "learning_rate_group_{}".format(group_nb), lr, epoch
        )

    def training_metric(self, metric, value, epoch):
        self._tensorboard_writer.add_scalar(f"training/{metric}", value, epoch)

    def validation_result(self, metric, value, epoch):
        self._tensorboard_writer.add_scalar(f"validation/{metric}", value, epoch)


class VisdomVisualiser(Observer):
    def __init__(self, env_name, lr_groups_nb):
        self._vis = Visdom(env=env_name)
        self._windows = {
            "loss": self._create_plot_window(
                "#Epochs", "Losses", "Training and Validation Average Losses"
            ),
            "accuracy": self._create_plot_window(
                "#Epochs", "Accuracies", "Training and Validation Average Accuracies"
            ),
            "error": self._create_plot_window(
                "#Epochs", "Errors", "Training and Validation Average Errors"
            ),
            "precision": self._create_plot_window(
                "#Epochs", "Precisions", "Training and Validation Average Precisions"
            ),
            "recall": self._create_plot_window(
                "#Epochs", "Recalls", "Training and Validation Average Recalls"
            ),
        }
        if lr_groups_nb and lr_groups_nb > 1:
            for i in range(lr_groups_nb):
                self._windows[f"lr_group_{i}"] = self._create_plot_window(
                    "#Epochs",
                    f"Learning Rate Group {i}",
                    f"Learning rate of group {i} at each start of epoch",
                )
        else:
            self._windows["lr"] = self._create_plot_window(
                "#Epochs", "Learning Rate", "Learning rate at each start of epoch"
            )

    def lr(self, lr, epoch):
        self._vis.line(
            X=np.array([epoch]),
            Y=np.array([lr]),
            win=self._windows["lr"],
            update="append",
        )

    def lr_group(self, lr, group_nb, epoch):
        self._vis.line(
            X=np.array([epoch]),
            Y=np.array([lr]),
            win=self._windows[f"lr_group_{group_nb}"],
            update="append",
        )

    def training_metric(self, metric, value, epoch):
        self._vis.line(
            X=np.array([epoch]),
            Y=np.array([value]),
            win=self._windows[metric],
            name="training",
            update="append",
        )

    def validation_result(self, metric, value, epoch):
        self._vis.line(
            X=np.array([epoch]),
            Y=np.array([value]),
            win=self._windows[metric],
            name="validation",
            update="append",
        )

    def _create_plot_window(self, xlabel, ylabel, title):
        return self._vis.line(
            X=np.array([1]),
            Y=np.array([np.nan]),
            opts=dict(xlabel=xlabel, ylabel=ylabel, title=title),
        )
