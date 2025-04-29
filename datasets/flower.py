import cv2
import glob
import scipy.io

import torch.utils.data as data


# Oxford-102-flower
class Flower(data.Dataset):
    def __init__(
        self, image_shape, data_dir="/data/oxford_102_flower/", validation=False
    ):
        self.image_shape = image_shape
        self.splits = scipy.io.loadmat(data_dir + "setid.mat")
        self.labels = scipy.io.loadmat(data_dir + "imagelabels.mat")["labels"][0]
        self.image_path = data_dir + "images/"
        file_list = glob.glob(self.image_path + "*.jpg")
        if validation:
            split_set = set(self.splits["valid"][0])
        else:
            split_set = set(self.splits["trnid"][0])  # | set(self.splits["tstid"][0])
        self.file_list = list(
            filter(
                lambda fname: self.file_name_to_number(fname) in split_set,
                file_list,
            )
        )

    def file_name_to_number(self, fname):
        return int(fname.split("/")[-1].split(".")[0].split("_")[-1])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(self.file_list[index])
        if img.shape != self.image_shape:
            img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        number = self.file_name_to_number(self.file_list[index])
        label = self.labels[number - 1] - 1  # [1 ~ 102] to [0 ~ 101]
        return img / 255.0, label


if __name__ == "__main__":
    ds = Flower()
    for index in range(7000):
        image, label = ds[index]
        print("label:", label)
        print("image:", image.shape)
