import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
from .tinyimagenet_module import TinyImageNet
import numpy as np

def get_normalization(dataset_name):
    '''
    Use ImageNet numbers for TinyImageNet too, so we can use pre-trained models.
    '''
    if dataset_name == 'ImageNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset_name == 'TinyImageNet':
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
    elif dataset_name == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif dataset_name == 'CIFAR100':
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif dataset_name == 'SVHN':
        mean = [x / 255.0 for x in [109.9, 109.7, 113.8]]
        std = [x / 255.0 for x in [50.1, 50.6, 50.8]]
    return mean, std

def get_num_classes(dataset_name):
    if dataset_name == 'ImageNet':
        return 1000
    elif dataset_name == 'TinyImageNet':
        return 200
    elif dataset_name == 'CIFAR10':
        return 10
    elif dataset_name == 'CIFAR100':
        return 100
    elif dataset_name == 'SVHN':
        return 10
        
def get_dimensions(dataset_name):
    '''
    returns standard sizes.  For TinyImageNet, we resize from 64x64 to 224x224 to use imagenet models
    '''
    if dataset_name == 'ImageNet' or dataset_name == 'TinyImageNet':
        return (3, 224, 224)
    elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100' or dataset_name == 'SVHN':
        return (3, 32, 32)

def make_trainset(dataset_name, data_dir='~/data', imagenet_resize=False):
    '''
    When using CIFAR10 or CIFAR100 datasets, the data_dir should be the directory containing the respective CIFAR folder.
    When using ImageNet or TinyImageNet, the data_dir should be the directory containing 'train' and 'val folders.
    '''
    mean, std = get_normalization(dataset_name)
    normalize = transforms.Normalize(mean=mean, std=std)
    if dataset_name == 'ImageNet':
        traindir = os.path.join(data_dir, 'train')
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, transform_train)
    elif dataset_name == 'TinyImageNet':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = TinyImageNet(data_dir, split='train', transform=transform_train, in_memory=False)
    elif dataset_name == 'CIFAR10':
        if imagenet_resize:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])        
        train_dataset =  datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    elif dataset_name == 'CIFAR100':
        if imagenet_resize:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])          
        train_dataset =  datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    elif dataset_name == 'SVHN':
        if imagenet_resize:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                normalize,
            ])           
        train_dataset = datasets.SVHN(root=data_dir, split='train', transform=transform_train, download=True)
        # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
        #extra_dataset = datasets.SVHN(root=data_dir, split='extra', transform=transform_train, download=True)
        #train_dataset.data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        #train_dataset.labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)

    return train_dataset

def make_testset(dataset_name, data_dir='~/data', imagenet_resize=False):
    '''
    When using CIFAR10 or CIFAR100 datasets, the data_dir should be the directory containing the respective CIFAR folder.
    When using ImageNet, the data_dir should be the ImageNet directory containing 'train' and 'val folders.
    '''
    mean, std = get_normalization(dataset_name)
    normalize = transforms.Normalize(mean=mean, std=std)
    if dataset_name == 'ImageNet':
        valdir = os.path.join(data_dir, 'val')
        test_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    elif dataset_name == 'TinyImageNet':
        transform_test = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        test_dataset = TinyImageNet(data_dir, split='val', transform=transform_test, in_memory=False)
    elif dataset_name == 'CIFAR10':
        if imagenet_resize:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        test_dataset =  datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    elif dataset_name == 'CIFAR100':
        if imagenet_resize:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        test_dataset =  datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    elif dataset_name == 'SVHN':
        if imagenet_resize:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        test_dataset = datasets.SVHN(root=data_dir, split='test', transform=transform_test, download=True)
    return test_dataset
