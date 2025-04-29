"""Factory for models and datasets."""

import torch
import timm

from torchvision.models import resnet50
from datasets.mnist import MNIST
from datasets.flower import Flower


def create_resnet50(config):
    model = resnet50(pretrained=False)
    model.conv1 = torch.nn.Conv2d(
        config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Linear(model.fc.in_features, config.num_classes)
    return model


def create_vit(config):
    model = timm.create_model(
        "vit_small_patch32_224",
        pretrained=False,
        img_size=config.image_shape,
        patch_size=32,
        in_chans=config.channels,
        num_classes=config.num_classes,
    )
    return model


class ModelFactory:
    @classmethod
    def create_model(cls, model_name, config):
        model_map = {
            "resnet50": create_resnet50,
            "vit": create_vit,
        }
        fn = model_map[model_name]
        return fn(config)


class DatasetFactory:
    @classmethod
    def create_dataset(cls, dataset_name, config):
        dataset_map = {
            "mnist": (MNIST(), MNIST(validation=True)),
            "flower": (
                Flower(config.image_shape),
                Flower(config.image_shape, validation=True),
            ),
        }
        return dataset_map[dataset_name]
