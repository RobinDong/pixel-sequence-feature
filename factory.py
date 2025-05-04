"""Factory for models and datasets."""

import torch
import timm

from torchvision.models import resnet50
from datasets.mnist import MNIST
from datasets.flower import Flower
from datasets.imagenet100 import Imagenet100
from timm.models.vision_transformer import VisionTransformer


def load_partial_weights(model_new, model_pretrained):
    pretrained_dict = model_pretrained.state_dict()
    model_dict = model_new.state_dict()

    # Filter out incompatible keys (like patch embedding projection)
    compatible_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }

    model_dict.update(compatible_dict)
    model_new.load_state_dict(model_dict)
    return model_new


def interpolate_pos_embed(model, checkpoint_model):
    pos_embed_checkpoint = checkpoint_model.pos_embed
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches**0.5)

    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embedding_size).permute(
        0, 3, 1, 2
    )
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
        1, new_size * new_size, embedding_size
    )
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    model.pos_embed.data.copy_(new_pos_embed)


class ViTSmallPatch4(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=4,
            embed_dim=384,  # Same as vit_small
            depth=12,
            num_heads=6,
            **kwargs,
        )


class ViTSmallPatch2(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            patch_size=2,
            embed_dim=384,  # Same as vit_small
            depth=12,
            num_heads=6,
            **kwargs,
        )


def create_resnet50(config):
    model = resnet50(pretrained=False)
    model.conv1 = torch.nn.Conv2d(
        config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )
    model.fc = torch.nn.Linear(model.fc.in_features, config.num_classes)
    return model


def create_vit(config):
    if config.patch_size >= 8:
        model = timm.create_model(
            f"vit_small_patch{config.patch_size}_224",
            pretrained=True,
            img_size=config.image_shape,
            in_chans=config.channels,
            num_classes=config.num_classes,
        )
    else:
        model = timm.create_model(
            "vit_small_patch8_224",
            pretrained=True,
            img_size=config.image_shape,
            in_chans=config.channels,
            num_classes=config.num_classes,
        )
        if config.patch_size == 4:
            model_new = ViTSmallPatch4(
                img_size=config.image_shape, num_classes=config.num_classes
            )
        elif config.patch_size == 2:
            model_new = ViTSmallPatch2(
                img_size=config.image_shape, num_classes=config.num_classes
            )
        model_new = load_partial_weights(model_new, model)
        interpolate_pos_embed(model_new, model)
        model = model_new
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
                Flower(
                    config.image_shape,
                ),
                Flower(
                    config.image_shape,
                    validation=True,
                ),
            ),
            "imagenet100": (
                Imagenet100(
                    config.image_shape,
                ),
                Imagenet100(
                    config.image_shape,
                    validation=True,
                ),
            ),
        }
        return dataset_map[dataset_name]
