import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datasets import CIFAR10
from models import CNN_CIFAR10


def test_CIFAR10(dataset_name, model_name):
    BATCH_SIZE = 32
    DIR = "./workspace/"

    # 加载测试数据集
    if dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 标准化
        ])
        test_loader = DataLoader(CIFAR10(train=False, transform=transform), batch_size=BATCH_SIZE)
    if model_name == "CNN":
        model = CNN_CIFAR10()
        model.load_state_dict(torch.load(f'{DIR}{dataset_name}/{model_name}.pth'))

    # 测试模型
    model.eval()  # 将模型设为评估模式，不使用 dropout 等
    correct = 0
    total = 0

    with torch.no_grad():  # 在测试阶段不需要计算梯度
        for data in test_loader:
            inputs, labels = data

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {100 * accuracy:.2f}%")