from .dataset_torch import cifar_dataset_read, fashionmnist_dataset_read, eurosat_dataset_read, tiny_imagenet_dataset_read,get_all_pacs_dataloader,get_all_officehome_dataloader,get_all_officehome_dataloader_art,get_all_officehome_dataloader_full
from .dataset_leaf import leaf_read, sent140_read
from .dataset_torch import get_vlcs_dataloader,get_vlcs_dataloader_vcs
def get_dataloader(args):
    if args.dataset == 'sent140':
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions, sent140_info = sent140_read(args)
        return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions, sent140_info
    if args.dataset in ('cifar10', 'cifar100'):
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = cifar_dataset_read(args, args.dataset, args.datadir, args.batch_size, args.n_client, args.partition, args.beta, args.skew_class)
    elif args.dataset == 'fashionmnist':
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = fashionmnist_dataset_read(args, args.dataset, args.datadir, args.batch_size, args.n_client, args.partition, args.beta, args.skew_class)
    elif args.dataset in ('femnist', 'shakespeare'):
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = leaf_read(args)
    elif args.dataset == 'eurosat':
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = eurosat_dataset_read(args, args.dataset, args.datadir, args.batch_size, args.n_client, args.partition, args.beta, args.skew_class)
    elif args.dataset == 'tiny_imagenet':
        train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions = tiny_imagenet_dataset_read(args, args.dataset, args.datadir, args.batch_size, args.n_client, args.partition, args.beta, args.skew_class)
    elif args.dataset == 'pacs':
        train_dataloaders, test_dataloaders, client_num_samples, domain = get_all_pacs_dataloader(args,args.datadir,args.batch_size,args.n_client,args.partition, args.beta, args.skew_class)
        return train_dataloaders,test_dataloaders,client_num_samples,domain
    elif args.dataset == 'officehome':
        train_dataloaders, test_dataloaders, client_num_samples, domain = get_all_officehome_dataloader(args,args.datadir,args.batch_size,args.n_client,args.partition, args.beta, args.skew_class)
        return train_dataloaders,test_dataloaders,client_num_samples,domain
    elif args.dataset == 'vlcs':
        train_dataloaders, test_dataloaders, client_num_samples, domain = get_vlcs_dataloader(args,args.datadir,args.batch_size,args.n_client,args.partition, args.beta, args.skew_class)
        return train_dataloaders,test_dataloaders,client_num_samples,domain
    return train_dataloaders, test_loader, client_num_samples, traindata_cls_counts, data_distributions