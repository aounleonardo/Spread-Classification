from dataclasses import dataclass


@dataclass
class Configuration:
    dataset_path: str
    epochs: int
    output: str
    optimizer: str
    model_name: str
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    lr: int
    lr_update_every: int
    weight_decay: float
    gamma: float
    restart_every: int
    restart_factor: float
    init_lr_factor: float
    lr_reduce_patience: int
    lr_reduce_factor: float
    early_stop_patience: int
    debug: bool
    log_interval: int

    run_index: int
    log_dir: str
    log_level: int
    device: str
    dataset_size: int

    def __repr__(self):
        return f"""
            Training configuration:
                Model: {self.model_name}
                Train batch size: {self.train_batch_size}
                Val batch size: {self.val_batch_size}
                Dataset size: {self.dataset_size}
                Number of workers: {self.num_workers}
                Number of epochs: {self.epochs}
                Optimizer: {self.optimizer}
                Learning rate: {self.lr}
                Learning rate update every : {self.lr_update_every} epoch(s)
                Weigth decay rate: {self.weight_decay}
                Exp lr scheduler gamma: {self.gamma}
                    restart every: {self.restart_every}
                    restart factor: {self.restart_factor}
                    init lr factor: {self.init_lr_factor}
                Reduce on plateau:
                    patience: {self.lr_reduce_patience}
                    factor: {self.lr_reduce_factor}
                Early stopping patience: {self.early_stop_patience}
                Output folder: {self.output}
                Run Index: {self.run_index}
                Device: {self.device}
        """

    @classmethod
    def from_dict(cls, **kwargs):
        params = {
            param: value
            for param, value in kwargs.items()
            if param in cls.__annotations__
        }
        return cls(**params)
