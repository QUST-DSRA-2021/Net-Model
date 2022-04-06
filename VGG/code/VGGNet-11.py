import torch
from torch import nn


# 定义 VGG 块
def vgg_block(num_convs, in_channels, out_channels):
    # 创建层
    layers = []
    for _ in range(num_convs):
        # 卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        # ReLU层
        layers.append(nn.ReLU())
        in_channels = out_channels
    # 池化层
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    # 将创建的层放入 Sequential
    return nn.Sequential(*layers)


# VGG
class VGG(nn.Module):
    def __init__(self, conv_arch):
        super(VGG, self).__init__()

        conv_blocks = []
        in_channls = 1
        for num_convs, out_channels in conv_arch:
            conv_blocks.append(vgg_block(num_convs, in_channls, out_channels))
            in_channls = out_channels

        self.convs = nn.Sequential(*conv_blocks)
        self.fal = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.fal(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    net = VGG(((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))
    print(net)

