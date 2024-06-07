import torch.utils.data as data
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import ConcatDataset
import os
import sys

from .partition_data import partition_data
from .partition_data_gen import read_gen_data, distribute_data

class Cifar_Truncated(data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(Cifar_Truncated, self).__init__()
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

def cifar_dataset_read(args, dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    data_path = os.path.join(base_path, dataset)
    if dataset == "cifar10":
        train_dataset = CIFAR10(data_path, True, download=True)
        test_dataset = CIFAR10(data_path, False, download=True)
        if args.mia_transforms == True:
            transform_train=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform_train=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == "cifar100":
        train_dataset = CIFAR100(data_path, True, download=True)
        test_dataset = CIFAR100(data_path, False, download=True)
        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

        transform_test=transforms.Compose(
            [transforms.ToTensor(),
            normalize])
        
    train_image = train_dataset.data
    train_label = np.array(train_dataset.targets)
    test_image = test_dataset.data
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]

    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)

    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        local_train_image = train_image[train_idxs]
        local_train_label = train_label[train_idxs]
        train_dataset = Cifar_Truncated(data=local_train_image, labels=local_train_label, transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions


def eurosat_dataset_read(args, dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    data_path = os.path.join(base_path,dataset)
    data_path = os.path.join(data_path,'real_data')
    train_image = np.load(os.path.join(data_path, 'train_image.npy'))
    train_image = train_image.astype(np.uint8)
    train_label = np.load(os.path.join(data_path, 'train_label.npy'))
    train_label = train_label.astype(int)
    test_image = np.load(os.path.join(data_path, 'test_image.npy'))
    test_image = test_image.astype(np.uint8)
    test_label = np.load(os.path.join(data_path, 'test_label.npy'))
    test_label = test_label.astype(int)
    transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.37239245, 0.3723767, 0.37261853), (0.08610647, 0.08577179, 0.08604158))])
    transform_test=transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.37239245, 0.3723767, 0.37261853), (0.08610647, 0.08577179, 0.08604158))])
    n_train = train_label.shape[0]
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        local_train_image =  train_image[train_idxs]
        local_train_label =  train_label[train_idxs]
        train_dataset = Cifar_Truncated(data=local_train_image, labels=local_train_label, transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=bool(args.mia_drop_last))
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

