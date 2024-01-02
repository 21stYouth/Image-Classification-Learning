import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision

class CIFAR10(Dataset):
    def __init__(self, train, transform=None):
        self.transform = transform
        if train:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
        else:
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        if self.transform == None:
            image = transforms.ToTensor()(image)
        return image, target

