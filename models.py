import torch
import torch.nn as nn
from datasets import CIFAR10


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 10是CIFAR-10的类别数

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入 x 的形状应该是 (batch_size, channels, height, width)
        # 第一层卷积和激活函数
        x = self.relu(self.conv1(x))
        # 第一层池化
        x = self.pool(x)

        # 第二层卷积和激活函数
        x = self.relu(self.conv2(x))
        # 第二层池化
        x = self.pool(x)

        # 将特征图展平
        x = x.view(-1, 64 * 8 * 8)

        # 全连接层和激活函数
        x = self.relu(self.fc1(x))
        # 输出层
        x = self.fc2(x)

        return x
