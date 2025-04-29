from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_workers: int = 8
    lr: float = 1e-2
    batch_size: int = 64
    grad_clip: float = 0.5
    epochs: int = 35
    num_classes: int = 10
    channels: int = 1
    image_shape: tuple[int] = (28, 28)


def get_config(dataset_name):
    config_map = {
        "mnist": TrainConfig(
            lr=1e-2, batch_size=64, num_classes=10, channels=1, image_shape=(28, 28)
        ),
        "flower": TrainConfig(
            lr=1e-4, batch_size=16, num_classes=102, channels=3, image_shape=(500, 666)
        ),
    }
    return config_map[dataset_name]
