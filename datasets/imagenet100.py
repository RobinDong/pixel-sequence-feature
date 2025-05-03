import cv2
import glob
import json
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


# Imagenet-100
class Imagenet100(data.Dataset):
    def __init__(
        self, image_shape=(224, 224), data_dir="/data/imagenet100", validation=False
    ):
        self.image_shape = image_shape
        if validation:
            self.file_list = glob.glob(f"{data_dir}/val.*/*/*.JPEG")
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.file_list = glob.glob(f"{data_dir}/train.*/*/*.JPEG")
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    AutoAugment(
                        policy=AutoAugmentPolicy.IMAGENET
                    ),  # <- AutoAugment step
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        with open(f"{data_dir}/Labels.json", "r") as fp:
            content = json.load(fp)

        id_map = {}
        self.dir_to_id = {}
        id = 0
        for dir_name, real_name in content.items():
            self.dir_to_id[dir_name] = id
            id_map[id] = real_name
            id += 1

        # write down (id -> real_name)
        with open("id_map.json", "w") as fp:
            json.dump(id_map, fp)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index])
        if img.shape != self.image_shape:
            img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        dir_name = self.file_list[index].split("/")[-2]
        label = self.dir_to_id[dir_name]
        img = self.transform(Image.fromarray(img))
        return img, label


if __name__ == "__main__":
    ds = Imagenet100()
    print("length:", len(ds))
    for index in (5, 7, 9, 131, 250, 397):
        image, label = ds[index]
        print("label:", label)
        print("image:", image.shape)
