import albumentations as A
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(root, augmentations=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    
    if augmentations is not None:
        transform = A.Compose(augmentations + [transform])
        
    testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainset, testset, classes
