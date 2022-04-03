import os
import gzip
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset


# 读取本地数据
def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    return x_train, y_train


class MyDataset(Dataset):
    """
    用于读取和初始化数据
    """
    # 读取数据文件
    def __init__(self, folder, data_name, label_name, transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name)
        self.train_data = train_set
        self.train_labels = train_labels
        self.transform = transform

    # 进行支持下标访问
    def __getitem__(self, index):
        img, labels = self.train_data[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)

        return img, labels

    # 返回自定义数据集的大小，方便后期遍历
    def __len__(self):

        return len(self.train_data)


if __name__ == "__main__":
    # 使用步骤说明
    # 一、先从网上下载数据
    train_dataset = datasets.MNIST(root='../', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='../', train=False, transform=transforms.ToTensor(), download=True)

    # 二、本地读取数据（下载之后无需再进行步骤一）
    trainDataset = MyDataset('../MNIST/raw', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                             transform=transforms.ToTensor())
    # 三、训练数据的装载
    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=10,
        shuffle=False,
    )







