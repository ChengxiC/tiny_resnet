# Chengxi Chu, Universiti Malaya
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models


train_cifar = datasets.CIFAR10(
    root='./data/',
    train=True,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
    download=True,
)

loader_train_cifar = DataLoader(
    dataset=train_cifar,
    batch_size=32,
    shuffle=True,
    drop_last=False,
)

test_cifar = datasets.CIFAR10(
    root='./data/',
    train=False,
    transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
    download=True,
)

loader_test_cifar = DataLoader(
    dataset=test_cifar,
    batch_size=32,
    shuffle=True,
    drop_last=False,
)


x, labels = iter(loader_train_cifar).__next__()
print(f'x:{x.shape}; labels:{labels.shape}')  # x:torch.Size([32, 3, 32, 32]); labels:torch.Size([32])








