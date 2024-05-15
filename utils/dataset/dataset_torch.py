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
    if args.mia_small_train == True:
        n_train = int(n_train*args.mia_small_fraction)
        train_image = train_image[:n_train]
        train_label = train_label[:n_train]
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)

    # ===== Include generated data =====
    if args.fedgc:
        gen_image, gen_label = read_gen_data(args, os.path.join(data_path, f'generation_{args.fedgc_generator}'))
        gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
        client_num_samples = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples   # aggregated according to new dataset size

    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        local_train_image = np.concatenate((train_image[train_idxs], gen_image[gen_idxs[i]])) if args.fedgc else train_image[train_idxs]
        local_train_label = np.hstack((train_label[train_idxs], gen_label[gen_idxs[i]])) if args.fedgc else train_label[train_idxs]
        train_dataset = Cifar_Truncated(data=local_train_image, labels=local_train_label, transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=bool(args.mia_drop_last))
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

def fashionmnist_dataset_read(args, dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    data_path = os.path.join(base_path, dataset)
    train_dataset = FashionMNIST(data_path, True, download=True)
    test_dataset = FashionMNIST(data_path, False, download=True)
    transform_train=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])

    transform_test=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
        
    train_image = np.array(train_dataset.data)
    train_label = np.array(train_dataset.targets)
    test_image = np.array(test_dataset.data)
    test_label = np.array(test_dataset.targets)
    n_train = train_label.shape[0]
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        train_dataset = Cifar_Truncated(data=train_image[train_idxs], labels=train_label[train_idxs], transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

def eurosat_dataset_read(args, dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    data_path = os.path.join(base_path,dataset)
    data_path = os.path.join(data_path,'real_data')
    # images = np.load(os.path.join(data_path, 'images.npy'))
    # labels = np.load(os.path.join(data_path, 'labels.npy'))
    # train_image = np.array([])
    # train_label = np.array([])
    # test_image = np.array([])
    # test_label = np.array([])
    # total_count = np.zeros(10)
    # train_test_boundary = np.array([2400, 2400, 2400, 2000, 2000, 1600, 2000, 2400, 2000, 2400])
    # for i in range(27000):
    #     print('processing image_id: ',i)
    #     if total_count[labels[i]]<train_test_boundary[labels[i]]:
    #         train_image = np.append(train_image,images[i])
    #         train_label = np.append(train_label,labels[i])
    #         total_count[labels[i]] = total_count[labels[i]]+1
    #     else:
    #         test_image = np.append(test_image,images[i])
    #         test_label = np.append(test_label,labels[i])
    # train_image = train_image.reshape(-1,64,64,3)
    # test_image = test_image.reshape(-1,64,64,3)
    train_image = np.load(os.path.join(data_path, 'train_image.npy'))
    # print(train_image.shape)
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
    if args.fedgc:
        eurosat_path = os.path.join(base_path,dataset)
        gen_image, gen_label = read_gen_data(args, os.path.join(eurosat_path, f'generation_{args.fedgc_generator}'))
        gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
        client_num_samples = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples   # aggregated according to new dataset size
    
    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        local_train_image = np.concatenate((train_image[train_idxs], gen_image[gen_idxs[i]])) if args.fedgc else train_image[train_idxs]
        local_train_label = np.hstack((train_label[train_idxs], gen_label[gen_idxs[i]])) if args.fedgc else train_label[train_idxs]
        train_dataset = Cifar_Truncated(data=local_train_image, labels=local_train_label, transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=bool(args.mia_drop_last))
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions

def tiny_imagenet_dataset_read(args, dataset, base_path, batch_size, n_parties, partition, beta, skew_class):
    data_path = os.path.join(base_path, dataset)
    data_path = os.path.join(data_path,'tiny-imagenet-200')
    transform_train=transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    # transform_test=transforms.Compose(
    #         [transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_image = torch.load(os.path.join(data_path,'train_image.pt'))
    train_label = torch.load(os.path.join(data_path,'train_label.pt'))
    train_label = np.array(train_label)
    train_label = train_label.astype(int)
    test_image = torch.load(os.path.join(data_path,'test_image.pt'))
    test_label = torch.load(os.path.join(data_path,'test_label.pt'))
    test_label = np.array(test_label)
    test_label = test_label.astype(int)
    n_train = train_label.shape[0]
    net_dataidx_map, client_num_samples, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    if args.fedgc:
        tiny_imagenet_path = data_path
        gen_image, gen_label = read_gen_data(args, os.path.join(tiny_imagenet_path, f'generation_{args.fedgc_generator}'))
        gen_image = torch.tensor(gen_image)
        gen_image = gen_image/255
        gen_image = gen_image.permute(0,3,1,2)
        gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
        client_num_samples = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples   # aggregated according to new dataset size
    
    train_dataloaders = []
    for i in range(n_parties):
        train_idxs = net_dataidx_map[i]
        # print('np.concatenate')
        local_train_image = np.concatenate((train_image[train_idxs], gen_image[gen_idxs[i]])) if args.fedgc else train_image[train_idxs]
        if args.fedgc:
            local_train_image = torch.tensor(local_train_image) 
        # print(local_train_image.shape)
        # print(local_train_image)
        # print('np.nstack')
        local_train_label = np.hstack((train_label[train_idxs], gen_label[gen_idxs[i]])) if args.fedgc else train_label[train_idxs]
        # print('np finish')
        train_dataset = Cifar_Truncated(data=local_train_image, labels=local_train_label, transform=transform_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=bool(args.mia_drop_last))
        train_dataloaders.append(train_loader)
    
    test_dataset = Cifar_Truncated(data=test_image, labels=test_label, transform=transform_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions
    

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.transform = transform

        imagefolder_obj = ImageFolder(self.root, self.transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)
        
def PACS_domain_dataset_read(args, domain_name, base_path, batch_size, n_parties, partition, beta, skew_class):
    
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    real_datadir = os.path.join(base_path,'PACS')
    dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    labels = dataset_all.targets
    dataset_size = len(labels)
    idxs = np.random.permutation(dataset_size)
    train_idxs = idxs[:int(0.8*dataset_size)]
    test_idxs = idxs[int(0.8*dataset_size):]
    test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=test_idxs, transform=transforms_test)
    train_label = np.array(dataset_all.targets)[train_idxs]
    n_train = len(train_idxs)
    net_dataidx_map, client_num_samples_per_domain, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    
    if args.fedgc:
        gen_path = os.path.join(base_path, f'generation_{args.fedgc_generator}')
        # gen_image, gen_label = read_gen_data(args, os.path.join(base_path, f'generation_{args.fedgc_generator}'))
        gen_image, gen_label = read_gen_data(args, os.path.join(gen_path, domain_name))
        # gen_image = torch.tensor(gen_image)
        # gen_image = gen_image/255
        # gen_image = gen_image.permute(0,3,1,2)
        gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
        client_num_samples_per_domain = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples_per_domain   # aggregated according to new dataset size    
    
    train_dataloaders = []
    for i in range(n_parties):
        idxs = net_dataidx_map[i]
        train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=train_idxs[idxs], transform=transforms_train)
        if args.fedgc:
            train_dataset_gen = ImageFolder_custom(root=os.path.join(gen_path, domain_name),dataidxs=gen_idxs[i],transform=transforms_train)
            train_dataset = ConcatDataset([train_dataset,train_dataset_gen])
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
        train_dataloaders.append(train_loader)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False)
    return train_dataloaders, test_loader, client_num_samples_per_domain, traindata_cls_counts, data_distributions

def get_all_PACS_dloader(args,base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    base_path = os.path.join(base_path, "pacs")
    for domain in ["art_painting", "cartoon", "photo", "sketch"]:
        domain_train_dataloaders, domain_test_loader, client_num_samples_per_domain, _, _ = PACS_domain_dataset_read(args,domain, base_path, batch_size, n_parties, partition, beta, skew_class)
        for i in range(len(domain_train_dataloaders)):
            train_dataloaders.append(domain_train_dataloaders[i])
            domains.append(domain)
            client_num_samples.append(client_num_samples_per_domain[i])
        test_dataloaders.append(domain_test_loader)
    return train_dataloaders, test_dataloaders, np.array(client_num_samples), domains

def get_all_pacs_dataloader(args,base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    base_path = os.path.join(base_path, "pacs")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # art_painting
    domain_name = 'art_painting'
    real_datadir = os.path.join(base_path,'PACS')
    art_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    art_labels = art_dataset_all.targets
    art_dataset_size = len(art_labels)
    art_idxs = np.random.permutation(art_dataset_size)
    art_train_idxs = art_idxs[:int(0.8*art_dataset_size)]
    art_test_idxs = art_idxs[int(0.8*art_dataset_size):]
    art_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_test_idxs, transform=transforms_test)
    art_train_label = np.array(art_dataset_all.targets)[art_train_idxs]
    art_n_train = len(art_train_idxs)
    art_train_dataloaders = []
    art_net_dataidx_map, art_client_num_samples_per_domain, art_traindata_cls_counts, art_data_distributions = partition_data(partition, art_n_train, n_parties, art_train_label, beta, skew_class)
    art_train_datasets = []
    for i in range(n_parties):
        art_idxs = art_net_dataidx_map[i]
        art_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_train_idxs[art_idxs], transform=transforms_train)
        art_train_datasets.append(art_train_dataset)
        client_num_samples.append(art_client_num_samples_per_domain[i])
    art_test_loader = DataLoader(art_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False)
    test_dataloaders.append(art_test_loader)
    
    # cartoon
    domain_name = 'cartoon'
    real_datadir = os.path.join(base_path,'PACS')
    cartoon_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    cartoon_labels = cartoon_dataset_all.targets
    cartoon_dataset_size = len(cartoon_labels)
    cartoon_idxs = np.random.permutation(cartoon_dataset_size)
    cartoon_train_idxs = cartoon_idxs[:int(0.8*cartoon_dataset_size)]
    cartoon_test_idxs = cartoon_idxs[int(0.8*cartoon_dataset_size):]
    cartoon_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=cartoon_test_idxs, transform=transforms_test)
    cartoon_train_label = np.array(cartoon_dataset_all.targets)[cartoon_train_idxs]
    cartoon_n_train = len(cartoon_train_idxs)
    cartoon_train_dataloaders = []
    cartoon_net_dataidx_map, cartoon_client_num_samples_per_domain, cartoon_traindata_cls_counts, cartoon_data_distributions = partition_data(partition, cartoon_n_train, n_parties, cartoon_train_label, beta, skew_class)
    cartoon_train_datasets = []
    for i in range(n_parties):
        cartoon_idxs = cartoon_net_dataidx_map[i]
        cartoon_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=cartoon_train_idxs[cartoon_idxs], transform=transforms_train)
        cartoon_train_datasets.append(cartoon_train_dataset)
        client_num_samples.append(cartoon_client_num_samples_per_domain[i])
    cartoon_test_loader = DataLoader(cartoon_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(cartoon_test_loader)

    #photo
    domain_name = 'photo'
    real_datadir = os.path.join(base_path,'PACS')
    photo_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    photo_labels = photo_dataset_all.targets
    photo_dataset_size = len(photo_labels)
    photo_idxs = np.random.permutation(photo_dataset_size)
    photo_train_idxs = photo_idxs[:int(0.8*photo_dataset_size)]
    photo_test_idxs = photo_idxs[int(0.8*photo_dataset_size):]
    photo_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=photo_test_idxs, transform=transforms_test)
    photo_train_label = np.array(photo_dataset_all.targets)[photo_train_idxs]
    photo_n_train = len(photo_train_idxs)
    photo_train_dataloaders = []
    photo_net_dataidx_map, photo_client_num_samples_per_domain, photo_traindata_cls_counts, photo_data_distributions = partition_data(partition, photo_n_train, n_parties, photo_train_label, beta, skew_class)
    photo_train_datasets = []
    for i in range(n_parties):
        photo_idxs = photo_net_dataidx_map[i]
        photo_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=photo_train_idxs[photo_idxs], transform=transforms_train)
        photo_train_datasets.append(photo_train_dataset)
        client_num_samples.append(photo_client_num_samples_per_domain[i])
    photo_test_loader = DataLoader(photo_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(photo_test_loader)
    
    #sketch
    domain_name = 'sketch'
    real_datadir = os.path.join(base_path,'PACS')
    sketch_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    sketch_labels = sketch_dataset_all.targets
    sketch_dataset_size = len(sketch_labels)
    sketch_idxs = np.random.permutation(sketch_dataset_size)
    sketch_train_idxs = sketch_idxs[:int(0.8*sketch_dataset_size)]
    sketch_test_idxs = sketch_idxs[int(0.8*sketch_dataset_size):]
    sketch_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=sketch_test_idxs, transform=transforms_test)
    sketch_train_label = np.array(sketch_dataset_all.targets)[sketch_train_idxs]
    sketch_n_train = len(sketch_train_idxs)
    sketch_train_dataloaders = []
    sketch_net_dataidx_map, sketch_client_num_samples_per_domain, sketch_traindata_cls_counts, sketch_data_distributions = partition_data(partition, sketch_n_train, n_parties, sketch_train_label, beta, skew_class)
    sketch_train_datasets = []
    for i in range(n_parties):
        sketch_idxs = sketch_net_dataidx_map[i]
        sketch_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=sketch_train_idxs[sketch_idxs], transform=transforms_train)
        sketch_train_datasets.append(sketch_train_dataset)
        client_num_samples.append(sketch_client_num_samples_per_domain[i])
    sketch_test_loader = DataLoader(sketch_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(sketch_test_loader)

    for tmp_domain in ["art_painting", "cartoon", "photo", "sketch"]:
        if tmp_domain == "art_painting":
            traindata_cls_counts = art_traindata_cls_counts
            train_datasets = art_train_datasets
            client_num_samples_per_domain = art_client_num_samples_per_domain
        elif tmp_domain == "cartoon":
            traindata_cls_counts = cartoon_traindata_cls_counts
            train_datasets = cartoon_train_datasets
            client_num_samples_per_domain = cartoon_client_num_samples_per_domain
        elif tmp_domain == "photo":
            traindata_cls_counts = photo_traindata_cls_counts
            train_datasets = photo_train_datasets
            client_num_samples_per_domain = photo_client_num_samples_per_domain
        elif tmp_domain == "sketch":
            traindata_cls_counts = sketch_traindata_cls_counts
            train_datasets = sketch_train_datasets
            client_num_samples_per_domain = sketch_client_num_samples_per_domain
        if args.fedgc:
            gen_path = os.path.join(base_path, f'generation_{args.fedgc_generator}')
            # gen_image, gen_label = read_gen_data(args, os.path.join(base_path, f'generation_{args.fedgc_generator}'))
            gen_image, gen_label = read_gen_data(args, os.path.join(gen_path, tmp_domain))
            # gen_image = torch.tensor(gen_image)
            # gen_image = gen_image/255
            # gen_image = gen_image.permute(0,3,1,2)
            gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
            client_num_samples_per_domain = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples_per_domain   # aggregated according to new dataset size
            for i in range(n_parties):
                train_dataset_gen = ImageFolder_custom(root=os.path.join(gen_path, domain_name),dataidxs=gen_idxs[i],transform=transforms_train)
                train_dataset = ConcatDataset([train_datasets[i],train_dataset_gen])
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)
        else:
            for i in range(n_parties):
                train_loader = DataLoader(dataset=train_datasets[i], batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)


    return train_dataloaders, test_dataloaders, np.array(client_num_samples), domains    
    
def get_all_officehome_dataloader(args,base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    art_train_sum =np.zeros(65,dtype=int)
    art_test_sum =np.zeros(65,dtype=int)
    base_path = os.path.join(base_path, "officehome")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # Art
    domain_name = 'Art'
    real_datadir = os.path.join(base_path,'OfficeHomeDataset_10072016')
    art_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    art_labels = art_dataset_all.targets
    art_dataset_size = len(art_labels)
    art_idxs = np.random.permutation(art_dataset_size)
    art_train_idxs = art_idxs[:int(0.8*art_dataset_size)]
    art_test_idxs = art_idxs[int(0.8*art_dataset_size):]
    art_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_test_idxs, transform=transforms_test)
    art_train_label = np.array(art_dataset_all.targets)[art_train_idxs]
    art_test_label = np.array(art_dataset_all.targets)[art_test_idxs]
    for aa in art_train_label:
        art_train_sum[aa] +=1
    for aa in art_test_label:
        art_test_sum[aa] +=1
    print(art_train_sum)
    print(art_test_sum)
    art_n_train = len(art_train_idxs)
    art_train_dataloaders = []
    art_net_dataidx_map, art_client_num_samples_per_domain, art_traindata_cls_counts, art_data_distributions = partition_data(partition, art_n_train, n_parties, art_train_label, beta, skew_class)
    art_train_datasets = []
    for i in range(n_parties):
        art_idxs = art_net_dataidx_map[i]
        art_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_train_idxs[art_idxs], transform=transforms_train)
        art_train_datasets.append(art_train_dataset)
        client_num_samples.append(art_client_num_samples_per_domain[i])
    art_test_loader = DataLoader(art_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False)
    test_dataloaders.append(art_test_loader)
    
    # Clipart
    domain_name = 'Clipart'
    real_datadir = os.path.join(base_path,'OfficeHomeDataset_10072016')
    clipart_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    clipart_labels = clipart_dataset_all.targets
    clipart_dataset_size = len(clipart_labels)
    clipart_idxs = np.random.permutation(clipart_dataset_size)
    clipart_train_idxs = clipart_idxs[:int(0.8*clipart_dataset_size)]
    clipart_test_idxs = clipart_idxs[int(0.8*clipart_dataset_size):]
    clipart_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=clipart_test_idxs, transform=transforms_test)
    clipart_train_label = np.array(clipart_dataset_all.targets)[clipart_train_idxs]
    clipart_n_train = len(clipart_train_idxs)
    clipart_train_dataloaders = []
    clipart_net_dataidx_map, clipart_client_num_samples_per_domain, clipart_traindata_cls_counts, clipart_data_distributions = partition_data(partition, clipart_n_train, n_parties, clipart_train_label, beta, skew_class)
    clipart_train_datasets = []
    for i in range(n_parties):
        clipart_idxs = clipart_net_dataidx_map[i]
        clipart_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=clipart_train_idxs[clipart_idxs], transform=transforms_train)
        clipart_train_datasets.append(clipart_train_dataset)
        client_num_samples.append(clipart_client_num_samples_per_domain[i])
    clipart_test_loader = DataLoader(clipart_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(clipart_test_loader)

    #Product
    domain_name = 'Product'
    real_datadir = os.path.join(base_path,'OfficeHomeDataset_10072016')
    product_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    product_labels = product_dataset_all.targets
    product_dataset_size = len(product_labels)
    product_idxs = np.random.permutation(product_dataset_size)
    product_train_idxs = product_idxs[:int(0.8*product_dataset_size)]
    product_test_idxs = product_idxs[int(0.8*product_dataset_size):]
    product_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=product_test_idxs, transform=transforms_test)
    product_train_label = np.array(product_dataset_all.targets)[product_train_idxs]
    product_n_train = len(product_train_idxs)
    product_train_dataloaders = []
    product_net_dataidx_map, product_client_num_samples_per_domain, product_traindata_cls_counts, product_data_distributions = partition_data(partition, product_n_train, n_parties, product_train_label, beta, skew_class)
    product_train_datasets = []
    for i in range(n_parties):
        product_idxs = product_net_dataidx_map[i]
        product_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=product_train_idxs[product_idxs], transform=transforms_train)
        product_train_datasets.append(product_train_dataset)
        client_num_samples.append(product_client_num_samples_per_domain[i])
    product_test_loader = DataLoader(product_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(product_test_loader)
    
    #Real World
    domain_name = 'Real World'
    real_datadir = os.path.join(base_path,'OfficeHomeDataset_10072016')
    real_world_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    real_world_labels = real_world_dataset_all.targets
    real_world_dataset_size = len(real_world_labels)
    real_world_idxs = np.random.permutation(real_world_dataset_size)
    real_world_train_idxs = real_world_idxs[:int(0.8*real_world_dataset_size)]
    real_world_test_idxs = real_world_idxs[int(0.8*real_world_dataset_size):]
    real_world_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=real_world_test_idxs, transform=transforms_test)
    real_world_train_label = np.array(real_world_dataset_all.targets)[real_world_train_idxs]
    real_world_n_train = len(real_world_train_idxs)
    real_world_train_dataloaders = []
    real_world_net_dataidx_map, real_world_client_num_samples_per_domain, real_world_traindata_cls_counts, real_world_data_distributions = partition_data(partition, real_world_n_train, n_parties, real_world_train_label, beta, skew_class)
    real_world_train_datasets = []
    for i in range(n_parties):
        real_world_idxs = real_world_net_dataidx_map[i]
        real_world_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=real_world_train_idxs[real_world_idxs], transform=transforms_train)
        real_world_train_datasets.append(real_world_train_dataset)
        client_num_samples.append(real_world_client_num_samples_per_domain[i])
    real_world_test_loader = DataLoader(real_world_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(real_world_test_loader)

    for tmp_domain in ["Art", "Clipart", "Product", "Real World"]:
        if tmp_domain == "Art":
            traindata_cls_counts = art_traindata_cls_counts
            train_datasets = art_train_datasets
            client_num_samples_per_domain = art_client_num_samples_per_domain
        elif tmp_domain == "Clipart":
            traindata_cls_counts = clipart_traindata_cls_counts
            train_datasets = clipart_train_datasets
            client_num_samples_per_domain = clipart_client_num_samples_per_domain
        elif tmp_domain == "Product":
            traindata_cls_counts = product_traindata_cls_counts
            train_datasets = product_train_datasets
            client_num_samples_per_domain = product_client_num_samples_per_domain
        elif tmp_domain == "Real World":
            traindata_cls_counts = real_world_traindata_cls_counts
            train_datasets = real_world_train_datasets
            client_num_samples_per_domain = real_world_client_num_samples_per_domain
        if args.fedgc:
            gen_path = os.path.join(base_path, f'generation_{args.fedgc_generator}')
            # gen_image, gen_label = read_gen_data(args, os.path.join(base_path, f'generation_{args.fedgc_generator}'))
            gen_image, gen_label = read_gen_data(args, os.path.join(gen_path, tmp_domain))
            # gen_image = torch.tensor(gen_image)
            # gen_image = gen_image/255
            # gen_image = gen_image.permute(0,3,1,2)
            gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
            client_num_samples_per_domain = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples_per_domain   # aggregated according to new dataset size
            for i in range(n_parties):
                train_dataset_gen = ImageFolder_custom(root=os.path.join(gen_path, domain_name),dataidxs=gen_idxs[i],transform=transforms_train)
                train_dataset = ConcatDataset([train_datasets[i],train_dataset_gen])
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)
        else:
            for i in range(n_parties):
                train_loader = DataLoader(dataset=train_datasets[i], batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)


    return train_dataloaders, test_dataloaders, np.array(client_num_samples), domains 

def get_all_officehome_dataloader_art(args,base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    art_train_sum =np.zeros(65,dtype=int)
    art_test_sum =np.zeros(65,dtype=int)
    base_path = os.path.join(base_path, "officehome")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # Art
    domain_name = 'Art'
    real_datadir = os.path.join(base_path,'OfficeHomeDataset_10072016')
    art_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    art_labels = art_dataset_all.targets
    art_dataset_size = len(art_labels)
    art_idxs = np.random.permutation(art_dataset_size)
    art_train_idxs = art_idxs[:int(0.8*art_dataset_size)]
    art_test_idxs = art_idxs[int(0.8*art_dataset_size):]
    art_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_test_idxs, transform=transforms_test)
    art_train_label = np.array(art_dataset_all.targets)[art_train_idxs]
    art_test_label = np.array(art_dataset_all.targets)[art_test_idxs]
    for aa in art_train_label:
        art_train_sum[aa] +=1
    for aa in art_test_label:
        art_test_sum[aa] +=1
    print(art_train_sum)
    print(art_test_sum)
    art_n_train = len(art_train_idxs)
    art_train_dataloaders = []
    art_net_dataidx_map, art_client_num_samples_per_domain, art_traindata_cls_counts, art_data_distributions = partition_data(partition, art_n_train, n_parties, art_train_label, beta, skew_class)
    art_train_datasets = []
    for i in range(n_parties):
        art_idxs = art_net_dataidx_map[i]
        art_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_train_idxs[art_idxs], transform=transforms_train)
        art_train_datasets.append(art_train_dataset)
        client_num_samples.append(art_client_num_samples_per_domain[i])
    art_test_loader = DataLoader(art_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False)
    test_dataloaders.append(art_test_loader)

    for tmp_domain in ["Art"]:
        if tmp_domain == "Art":
            traindata_cls_counts = art_traindata_cls_counts
            train_datasets = art_train_datasets
            client_num_samples_per_domain = art_client_num_samples_per_domain

        if args.fedgc:
            gen_path = os.path.join(base_path, f'generation_{args.fedgc_generator}')
            # gen_image, gen_label = read_gen_data(args, os.path.join(base_path, f'generation_{args.fedgc_generator}'))
            gen_image, gen_label = read_gen_data(args, os.path.join(gen_path, tmp_domain))
            # gen_image = torch.tensor(gen_image)
            # gen_image = gen_image/255
            # gen_image = gen_image.permute(0,3,1,2)
            gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
            client_num_samples_per_domain = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples_per_domain   # aggregated according to new dataset size
            for i in range(n_parties):
                train_dataset_gen = ImageFolder_custom(root=os.path.join(gen_path, domain_name),dataidxs=gen_idxs[i],transform=transforms_train)
                train_dataset = ConcatDataset([train_datasets[i],train_dataset_gen])
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)
        else:
            for i in range(n_parties):
                train_loader = DataLoader(dataset=train_datasets[i], batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)   
    
    return train_dataloaders, test_dataloaders, np.array(client_num_samples), domains 

def get_all_officehome_dataloader_full(args,base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    train_sum =np.zeros(65,dtype=int)
    test_sum =np.zeros(65,dtype=int)
    base_path = os.path.join(base_path, "officehome")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # real full
    domain_name = 'Art'
    real_datadir = os.path.join(base_path,'full_real_data')
    dataset_all = ImageFolder(real_datadir, transforms_train)
    labels = dataset_all.targets
    dataset_size = len(labels)
    idxs = np.random.permutation(dataset_size)
    train_idxs = idxs[:int(0.8*dataset_size)]
    test_idxs = idxs[int(0.8*dataset_size):]
    test_dataset = ImageFolder_custom(root=real_datadir, dataidxs=test_idxs, transform=transforms_test)
    train_label = np.array(dataset_all.targets)[train_idxs]
    test_label = np.array(dataset_all.targets)[test_idxs]
    for aa in train_label:
        train_sum[aa] +=1
    for aa in test_label:
        test_sum[aa] +=1
    print(train_sum)
    print(test_sum)
    n_train = len(train_idxs)
    train_dataloaders = []
    net_dataidx_map, client_num_samples_per_domain, traindata_cls_counts, data_distributions = partition_data(partition, n_train, n_parties, train_label, beta, skew_class)
    train_datasets = []
    for i in range(n_parties):
        idxs = net_dataidx_map[i]
        train_dataset = ImageFolder_custom(root=real_datadir, dataidxs=train_idxs[idxs], transform=transforms_train)
        train_datasets.append(train_dataset)
        client_num_samples.append(client_num_samples_per_domain[i])
    art_test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False)
    test_dataloaders.append(art_test_loader)

    if args.fedgc:
        pass
    else:
        for i in range(n_parties):
            train_loader = DataLoader(dataset=train_datasets[i], batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
            train_dataloaders.append(train_loader)
  
    
    return train_dataloaders, test_dataloaders, np.array(client_num_samples), domains

def get_vlcs_dataloader(args,base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    base_path = os.path.join(base_path, "vlcs")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # art_painting
    domain_name = 'CALTECH'
    real_datadir = os.path.join(base_path,'vlcs_real')
    art_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    art_labels = art_dataset_all.targets
    art_dataset_size = len(art_labels)
    art_idxs = np.random.permutation(art_dataset_size)
    art_train_idxs = art_idxs[:int(0.8*art_dataset_size)]
    art_test_idxs = art_idxs[int(0.8*art_dataset_size):]
    art_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_test_idxs, transform=transforms_test)
    art_train_label = np.array(art_dataset_all.targets)[art_train_idxs]
    art_n_train = len(art_train_idxs)
    art_train_dataloaders = []
    art_net_dataidx_map, art_client_num_samples_per_domain, art_traindata_cls_counts, art_data_distributions = partition_data(partition, art_n_train, n_parties, art_train_label, beta, skew_class)
    art_train_datasets = []
    for i in range(n_parties):
        art_idxs = art_net_dataidx_map[i]
        art_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_train_idxs[art_idxs], transform=transforms_train)
        art_train_datasets.append(art_train_dataset)
        client_num_samples.append(art_client_num_samples_per_domain[i])
    art_test_loader = DataLoader(art_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False)
    test_dataloaders.append(art_test_loader)
    
    # cartoon
    domain_name = 'LABELME'
    real_datadir = os.path.join(base_path,'vlcs_real')
    cartoon_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    cartoon_labels = cartoon_dataset_all.targets
    cartoon_dataset_size = len(cartoon_labels)
    cartoon_idxs = np.random.permutation(cartoon_dataset_size)
    cartoon_train_idxs = cartoon_idxs[:int(0.8*cartoon_dataset_size)]
    cartoon_test_idxs = cartoon_idxs[int(0.8*cartoon_dataset_size):]
    cartoon_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=cartoon_test_idxs, transform=transforms_test)
    cartoon_train_label = np.array(cartoon_dataset_all.targets)[cartoon_train_idxs]
    cartoon_n_train = len(cartoon_train_idxs)
    cartoon_train_dataloaders = []
    cartoon_net_dataidx_map, cartoon_client_num_samples_per_domain, cartoon_traindata_cls_counts, cartoon_data_distributions = partition_data(partition, cartoon_n_train, n_parties, cartoon_train_label, beta, skew_class)
    cartoon_train_datasets = []
    for i in range(n_parties):
        cartoon_idxs = cartoon_net_dataidx_map[i]
        cartoon_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=cartoon_train_idxs[cartoon_idxs], transform=transforms_train)
        cartoon_train_datasets.append(cartoon_train_dataset)
        client_num_samples.append(cartoon_client_num_samples_per_domain[i])
    cartoon_test_loader = DataLoader(cartoon_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(cartoon_test_loader)

    #photo
    domain_name = 'PASCAL'
    real_datadir = os.path.join(base_path,'vlcs_real')
    photo_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    photo_labels = photo_dataset_all.targets
    photo_dataset_size = len(photo_labels)
    photo_idxs = np.random.permutation(photo_dataset_size)
    photo_train_idxs = photo_idxs[:int(0.8*photo_dataset_size)]
    photo_test_idxs = photo_idxs[int(0.8*photo_dataset_size):]
    photo_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=photo_test_idxs, transform=transforms_test)
    photo_train_label = np.array(photo_dataset_all.targets)[photo_train_idxs]
    photo_n_train = len(photo_train_idxs)
    photo_train_dataloaders = []
    photo_net_dataidx_map, photo_client_num_samples_per_domain, photo_traindata_cls_counts, photo_data_distributions = partition_data(partition, photo_n_train, n_parties, photo_train_label, beta, skew_class)
    photo_train_datasets = []
    for i in range(n_parties):
        photo_idxs = photo_net_dataidx_map[i]
        photo_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=photo_train_idxs[photo_idxs], transform=transforms_train)
        photo_train_datasets.append(photo_train_dataset)
        client_num_samples.append(photo_client_num_samples_per_domain[i])
    photo_test_loader = DataLoader(photo_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(photo_test_loader)
    
    #sketch
    domain_name = 'SUN'
    real_datadir = os.path.join(base_path,'vlcs_real')
    sketch_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    sketch_labels = sketch_dataset_all.targets
    sketch_dataset_size = len(sketch_labels)
    sketch_idxs = np.random.permutation(sketch_dataset_size)
    sketch_train_idxs = sketch_idxs[:int(0.8*sketch_dataset_size)]
    sketch_test_idxs = sketch_idxs[int(0.8*sketch_dataset_size):]
    sketch_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=sketch_test_idxs, transform=transforms_test)
    sketch_train_label = np.array(sketch_dataset_all.targets)[sketch_train_idxs]
    sketch_n_train = len(sketch_train_idxs)
    sketch_train_dataloaders = []
    sketch_net_dataidx_map, sketch_client_num_samples_per_domain, sketch_traindata_cls_counts, sketch_data_distributions = partition_data(partition, sketch_n_train, n_parties, sketch_train_label, beta, skew_class)
    sketch_train_datasets = []
    for i in range(n_parties):
        sketch_idxs = sketch_net_dataidx_map[i]
        sketch_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=sketch_train_idxs[sketch_idxs], transform=transforms_train)
        sketch_train_datasets.append(sketch_train_dataset)
        client_num_samples.append(sketch_client_num_samples_per_domain[i])
    sketch_test_loader = DataLoader(sketch_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(sketch_test_loader)

    for tmp_domain in ["CALTECH", "LABELME", "PASCAL", "SUN"]:
        if tmp_domain == "CALTECH":
            traindata_cls_counts = art_traindata_cls_counts
            train_datasets = art_train_datasets
            client_num_samples_per_domain = art_client_num_samples_per_domain
        elif tmp_domain == "LABELME":
            traindata_cls_counts = cartoon_traindata_cls_counts
            train_datasets = cartoon_train_datasets
            client_num_samples_per_domain = cartoon_client_num_samples_per_domain
        elif tmp_domain == "PASCAL":
            traindata_cls_counts = photo_traindata_cls_counts
            train_datasets = photo_train_datasets
            client_num_samples_per_domain = photo_client_num_samples_per_domain
        elif tmp_domain == "SUN":
            traindata_cls_counts = sketch_traindata_cls_counts
            train_datasets = sketch_train_datasets
            client_num_samples_per_domain = sketch_client_num_samples_per_domain
        if args.fedgc:
            gen_path = os.path.join(base_path, f'generation_{args.fedgc_generator}_domain')
            # gen_image, gen_label = read_gen_data(args, os.path.join(base_path, f'generation_{args.fedgc_generator}'))
            gen_image, gen_label = read_gen_data(args, os.path.join(gen_path, tmp_domain))
            # gen_image = torch.tensor(gen_image)
            # gen_image = gen_image/255
            # gen_image = gen_image.permute(0,3,1,2)
            gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
            client_num_samples_per_domain = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples_per_domain   # aggregated according to new dataset size
            for i in range(n_parties):
                train_dataset_gen = ImageFolder_custom(root=os.path.join(gen_path, domain_name),dataidxs=gen_idxs[i],transform=transforms_train)
                train_dataset = ConcatDataset([train_datasets[i],train_dataset_gen])
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)
        else:
            for i in range(n_parties):
                train_loader = DataLoader(dataset=train_datasets[i], batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)


    return train_dataloaders, test_dataloaders, np.array(client_num_samples), domains    
    pass

def get_vlcs_dataloader_vcs(args,base_path, batch_size, n_parties, partition, beta, skew_class):
    train_dataloaders = []
    test_dataloaders = []
    domains = []
    client_num_samples = []
    base_path = os.path.join(base_path, "vlcs")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    # art_painting
    domain_name = 'CALTECH'
    real_datadir = os.path.join(base_path,'vlcs_real')
    art_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    art_labels = art_dataset_all.targets
    art_dataset_size = len(art_labels)
    art_idxs = np.random.permutation(art_dataset_size)
    art_train_idxs = art_idxs[:int(0.8*art_dataset_size)]
    art_test_idxs = art_idxs[int(0.8*art_dataset_size):]
    art_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_test_idxs, transform=transforms_test)
    art_train_label = np.array(art_dataset_all.targets)[art_train_idxs]
    art_n_train = len(art_train_idxs)
    art_train_dataloaders = []
    art_net_dataidx_map, art_client_num_samples_per_domain, art_traindata_cls_counts, art_data_distributions = partition_data(partition, art_n_train, n_parties, art_train_label, beta, skew_class)
    art_train_datasets = []
    for i in range(n_parties):
        art_idxs = art_net_dataidx_map[i]
        art_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=art_train_idxs[art_idxs], transform=transforms_train)
        art_train_datasets.append(art_train_dataset)
        client_num_samples.append(art_client_num_samples_per_domain[i])
    art_test_loader = DataLoader(art_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False)
    test_dataloaders.append(art_test_loader)
    
    #photo
    domain_name = 'PASCAL'
    real_datadir = os.path.join(base_path,'vlcs_real')
    photo_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    photo_labels = photo_dataset_all.targets
    photo_dataset_size = len(photo_labels)
    photo_idxs = np.random.permutation(photo_dataset_size)
    photo_train_idxs = photo_idxs[:int(0.8*photo_dataset_size)]
    photo_test_idxs = photo_idxs[int(0.8*photo_dataset_size):]
    photo_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=photo_test_idxs, transform=transforms_test)
    photo_train_label = np.array(photo_dataset_all.targets)[photo_train_idxs]
    photo_n_train = len(photo_train_idxs)
    photo_train_dataloaders = []
    photo_net_dataidx_map, photo_client_num_samples_per_domain, photo_traindata_cls_counts, photo_data_distributions = partition_data(partition, photo_n_train, n_parties, photo_train_label, beta, skew_class)
    photo_train_datasets = []
    for i in range(n_parties):
        photo_idxs = photo_net_dataidx_map[i]
        photo_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=photo_train_idxs[photo_idxs], transform=transforms_train)
        photo_train_datasets.append(photo_train_dataset)
        client_num_samples.append(photo_client_num_samples_per_domain[i])
    photo_test_loader = DataLoader(photo_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(photo_test_loader)
    
    #sketch
    domain_name = 'SUN'
    real_datadir = os.path.join(base_path,'vlcs_real')
    sketch_dataset_all = ImageFolder(os.path.join(real_datadir, domain_name), transforms_train)
    sketch_labels = sketch_dataset_all.targets
    sketch_dataset_size = len(sketch_labels)
    sketch_idxs = np.random.permutation(sketch_dataset_size)
    sketch_train_idxs = sketch_idxs[:int(0.8*sketch_dataset_size)]
    sketch_test_idxs = sketch_idxs[int(0.8*sketch_dataset_size):]
    sketch_test_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=sketch_test_idxs, transform=transforms_test)
    sketch_train_label = np.array(sketch_dataset_all.targets)[sketch_train_idxs]
    sketch_n_train = len(sketch_train_idxs)
    sketch_train_dataloaders = []
    sketch_net_dataidx_map, sketch_client_num_samples_per_domain, sketch_traindata_cls_counts, sketch_data_distributions = partition_data(partition, sketch_n_train, n_parties, sketch_train_label, beta, skew_class)
    sketch_train_datasets = []
    for i in range(n_parties):
        sketch_idxs = sketch_net_dataidx_map[i]
        sketch_train_dataset = ImageFolder_custom(root=os.path.join(real_datadir, domain_name), dataidxs=sketch_train_idxs[sketch_idxs], transform=transforms_train)
        sketch_train_datasets.append(sketch_train_dataset)
        client_num_samples.append(sketch_client_num_samples_per_domain[i])
    sketch_test_loader = DataLoader(sketch_test_dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                              shuffle=False) 
    test_dataloaders.append(sketch_test_loader)

    for tmp_domain in ["CALTECH", "PASCAL", "SUN"]:
        if tmp_domain == "CALTECH":
            traindata_cls_counts = art_traindata_cls_counts
            train_datasets = art_train_datasets
            client_num_samples_per_domain = art_client_num_samples_per_domain
        elif tmp_domain == "PASCAL":
            traindata_cls_counts = photo_traindata_cls_counts
            train_datasets = photo_train_datasets
            client_num_samples_per_domain = photo_client_num_samples_per_domain
        elif tmp_domain == "SUN":
            traindata_cls_counts = sketch_traindata_cls_counts
            train_datasets = sketch_train_datasets
            client_num_samples_per_domain = sketch_client_num_samples_per_domain
        if args.fedgc:
            gen_path = os.path.join(base_path, f'generation_{args.fedgc_generator}_domain')
            # gen_image, gen_label = read_gen_data(args, os.path.join(base_path, f'generation_{args.fedgc_generator}'))
            gen_image, gen_label = read_gen_data(args, os.path.join(gen_path, tmp_domain))
            # gen_image = torch.tensor(gen_image)
            # gen_image = gen_image/255
            # gen_image = gen_image.permute(0,3,1,2)
            gen_idxs, client_num_samples_gen, traindata_cls_counts_gen = distribute_data(args, gen_label, traindata_cls_counts)
            client_num_samples_per_domain = client_num_samples_gen if args.fedgc_change_aggr else client_num_samples_per_domain   # aggregated according to new dataset size
            for i in range(n_parties):
                train_dataset_gen = ImageFolder_custom(root=os.path.join(gen_path, domain_name),dataidxs=gen_idxs[i],transform=transforms_train)
                train_dataset = ConcatDataset([train_datasets[i],train_dataset_gen])
                train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)
        else:
            for i in range(n_parties):
                train_loader = DataLoader(dataset=train_datasets[i], batch_size=batch_size, num_workers=0, pin_memory=True, shuffle=True)
                train_dataloaders.append(train_loader)
                domains.append(tmp_domain)


    return train_dataloaders, test_dataloaders, np.array(client_num_samples), domains    
    pass