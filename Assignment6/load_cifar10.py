import albumentations as A
import torch
import torchvision
import torchvision.transforms as transforms

augmentations = [A.HorizontalFlip(), A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                 A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16,
                                 min_width=16, fill_value=(0.5, 0.5, 0.5), mask_fill_value=None)]

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
