from dataclasses import dataclass


@dataclass
class TrainConfig:
    num_workers: int = 8
    lr: float = 1e-2
    batch_size: int = 64
    grad_clip: float = 0.5
    epochs: int = 365
    num_classes: int = 10
    channels: int = 1
    image_shape: tuple[int] = (28, 28)
    accumulation_steps: int = 4


def get_config(dataset_name):
    config_map = {
        "mnist": TrainConfig(
            lr=1e-2, batch_size=64, num_classes=10, channels=1, image_shape=(28, 28)
        ),
        "flower": TrainConfig(
            lr=1e-5,
            batch_size=8,
            num_classes=102,
            channels=3,
            image_shape=(224, 224),
            accumulation_steps=16,
        ),
        "imagenet100": TrainConfig(
            lr=1e-5,
            batch_size=128,
            num_classes=100,
            channels=3,
            image_shape=(224, 224),
            accumulation_steps=1,
        ),
    }
    return config_map[dataset_name]
