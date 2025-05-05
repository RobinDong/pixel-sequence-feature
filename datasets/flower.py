import cv2
import glob
import torch
import scipy.io
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


# Oxford-102-flower
class Flower(data.Dataset):
    def __init__(
        self,
        image_shape,
        data_dir="/data/oxford_102_flower/",
        sequenced=False,
        validation=False,
    ):
        if sequenced:
            data_dir = "/home/robin/code/Grounded-Segment-Anything/flower/"

        self.sequenced = sequenced
        self.image_shape = image_shape
        self.splits = scipy.io.loadmat(data_dir + "setid.mat")
        self.labels = scipy.io.loadmat(data_dir + "imagelabels.mat")["labels"][0]
        self.image_path = data_dir + "images/"
        file_list = glob.glob(self.image_path + "*.jpg")
        if validation:
            split_set = set(self.splits["valid"][0])
            ops = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        else:
            split_set = set(self.splits["trnid"][0])  # | set(self.splits["tstid"][0])
            ops = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # <- AutoAugment step
                transforms.ToTensor(),
            ]
        if not self.sequenced:
            ops += [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]
        self.transform = transforms.Compose(ops)
        self.file_list = list(
            filter(
                lambda fname: self.file_name_to_number(fname) in split_set,
                file_list,
            )
        )
        if not validation:
            self.file_list = self.file_list * 20

    def file_name_to_number(self, fname):
        return int(fname.split("/")[-1].split(".")[0].split("_")[-1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index])
        if img.shape != self.image_shape:
            img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        img = self.transform(Image.fromarray(img))
        number = self.file_name_to_number(self.file_list[index])
        label = self.labels[number - 1] - 1  # [1 ~ 102] to [0 ~ 101]

        if self.sequenced:
            sequence = []
            simg = img.permute(1, 2, 0)
            height, width, channel = simg.size()
            patch_size = 4
            cols = 224 // patch_size
            length = cols * cols
            zero_patch = torch.zeros(patch_size, patch_size, 3)
            for idx in range(0, height, patch_size):
                for jdx in range(0, width, patch_size):
                    patch = simg[idx : idx + patch_size, jdx : jdx + patch_size, :]
                    if torch.equal(patch, zero_patch):
                        continue
                    pos = (idx // patch_size * cols + jdx // patch_size) / length
                    item = torch.cat((patch.flatten() / 255.0, torch.tensor([pos])))
                    sequence.append(item)
            if len(sequence) <= 0:
                return None
            sequence = torch.stack(sequence)
            padded = torch.zeros(length, 49)
            padded[: sequence.shape[0]] = sequence
            return padded, label
        return img, label


if __name__ == "__main__":
    ds = Flower()
    for index in range(7000):
        image, label = ds[index]
        print("label:", label)
        print("image:", image.shape)
