import albumentations as A
import torch
import torchvision
import torchvision.transforms as transforms

def load_cifar10(root='./data', augmentations=None):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                           download=True, transform=transform)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    if augmentations is not None:
        augmentations = A.Compose(augmentations)
        
        trainset.train_data = np.array([augmentations(image=img)["image"] for img in trainset.train_data])
        trainset.train_data = trainset.train_data.astype(np.float32)
        trainset.train_data = torch.from_numpy(trainset.train_data)

    return trainset, testset, classes
