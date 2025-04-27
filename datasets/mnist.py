import struct
import numpy as np

import torch.utils.data as data


class MNIST(data.Dataset):
    def __init__(self, data_dir="./", validation=False):
        mnist_files = {
            "train": {
                "images": "train-images.idx3-ubyte",
                "labels": "train-labels.idx1-ubyte",
            },
            "val": {
                "images": "t10k-images.idx3-ubyte",
                "labels": "t10k-labels.idx1-ubyte",
            },
        }
        if validation:
            labels_fname = mnist_files["val"]["labels"]
            images_fname = mnist_files["val"]["images"]
        else:
            labels_fname = mnist_files["train"]["labels"]
            images_fname = mnist_files["train"]["images"]

        self.images = self.load_mnist_images(images_fname)
        self.labels = self.load_mnist_labels(labels_fname)

    def load_mnist_images(self, filename):
        with open(filename, "rb") as f:
            # Read the magic number (first 4 bytes)
            magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read the image data
            image_data = f.read()
            images = np.frombuffer(image_data, dtype=np.uint8)
            images = images.reshape((num_images, rows, cols))
            return images

    def load_mnist_labels(self, filename):
        with open(filename, "rb") as f:
            magic, num_labels = struct.unpack(">II", f.read(8))
            label_data = f.read()
            labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].astype("float32")
        label = self.labels[index]
        return image, label


if __name__ == "__main__":
    ds = MNIST("./")
    image, label = ds[1023]

    print("label:", label)

    # Let's show the image
    import matplotlib.pyplot as plt

    plt.imshow(image, cmap="gray")
    plt.title("First MNIST Image")
    plt.show()
