from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_workers: int = 8
    lr: float = 1e-2
    batch_size: int = 64
    grad_clip: float = 0.5
    epochs: int = 35
    num_classes: int = 10
