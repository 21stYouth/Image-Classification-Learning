import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets import CIFAR10
from models import CNN_CIFAR10


def train_CIFAR10(dataset_name, model_name):
    BATCH_SIZE = 32
    LR = 0.001
    EPOCH = 10
    DIR = "./workspace/"

    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomCrop(32, padding=4),  # 随机裁剪
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化
        ])
        train_loader = DataLoader(CIFAR10(train=True, transform=transform), batch_size=BATCH_SIZE)
    if model_name == "CNN":
        model = CNN_CIFAR10()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = LR)

    for epoch in range(EPOCH):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 200 == 199:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    print("Training finished")

    torch.save(model.state_dict(), f'{DIR}{dataset_name}/{model_name}.pth')
    print("Model parameters saved")


