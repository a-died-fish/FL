import os
from collections import defaultdict, OrderedDict
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from .utils_language import *

def read_json_files(data_dir, files):
    clients = []
    data = defaultdict(lambda: None)
    # files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])
    clients = list(data.keys())
    return clients, data

class FEMNIST():
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    def __init__(self, datadir, train_user_index, test_user_index, train=True, transform=None, target_transform=None):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        # train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/femnist/train",
        #                                                                          "./data/femnist/test")
        files = os.listdir(datadir)
        files = [f for f in files if f.endswith('.json')]
        
        clients, data = read_json_files(datadir, files)
        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for user in train_user_index:
                self.dic_users[user] = set()
                l = len(train_data_x)
                cur_x = data[clients[user]]['x']
                cur_y = data[clients[user]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[user].add(j + l)
                    train_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.target = train_data_y
        else:  # test就混杂在一起搞
            test_data_x = []
            test_data_y = []
            for user in test_user_index:
                cur_x = data[clients[user]]['x']
                cur_y = data[clients[user]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(np.array(cur_x[j]).reshape(28, 28))
                    test_data_y.append(cur_y[j])

            self.data = test_data_x
            self.target = test_data_y

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return torch.from_numpy((0.5-img)/0.5).float(), target

    def __len__(self):
        return len(self.data)
    
    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

class ShakeSpeare():
    def __init__(self, datadir, train_user_index, test_user_index, train=True):
        super(ShakeSpeare, self).__init__()
        # train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/shakespeare/train",
        #                                                                 "./data/shakespeare/test")
        self.train = train
        files = os.listdir(datadir)
        files = [f for f in files if f.endswith('.json')]
        
        clients, data = read_json_files(datadir, files)

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for user in train_user_index:
                self.dic_users[user] = set()
                l = len(train_data_x)
                cur_x = data[clients[user]]['x']
                cur_y = data[clients[user]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[user].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.target = train_data_y
        else:  # test就混杂在一起搞
            test_data_x = []
            test_data_y = []
            for i in test_user_index:   # test
                cur_x = data[clients[i]]['x']
                cur_y = data[clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])

            self.data = test_data_x
            self.target = test_data_y
        # letter_to_vec for each target
        self.target = [letter_to_vec(t) for t in self.target]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.target[index]
        indices = word_to_indices(sentence)
        # y = indices[1:].append(target)
        # target = indices[1:].append(target)
        indices = torch.LongTensor(np.array(indices))
        # y = torch.Tensor(np.array(y))
        # target = torch.LongTensor(np.array(target))
        return indices, target

    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

class Sent140(Dataset):
    def __init__(self, datadir, train_user_index, test_user_index,
                 tokenizer: Tokenizer = None, train=True):
        """get `Dataset` for sent140 dataset
        Args:
            data (list): sentence list data
            targets (list): next-character target list
            is_to_tokens (bool, optional), if tokenize data by using tokenizer
            tokenizer (Tokenizer, optional), tokenizer
        """
        self.datadir = datadir
        self.data_tokens_tensor = []
        self.targets_tensor = []
        self.vocab = None
        self.tokenizer = Tokenizer()
        self.fix_len = None
        self.train = train

        files = os.listdir(datadir)
        files = [f for f in files if f.endswith('.json')]

        if self.train:
            self.dic_users = {}
            clients, data_all = read_json_files(datadir, files)
            train_data_x = []
            train_data_y = []
            for user in train_user_index:  # 只有会用到的users
                # print(user)
                self.dic_users[user] = set()
                l = len(train_data_x)
                cur_x = data_all[clients[user]]['x']
                cur_y = data_all[clients[user]]['y']
                
                for j in range(len(cur_x)):
                    self.dic_users[user].add(j + l) 
                    train_data_x.append(self.tokenizer(cur_x[j]))
                    train_data_y.append(cur_y[j])
            self.data = train_data_x
            self.target = train_data_y
        else:  # test就混杂在一起搞
            clients, data_all = read_json_files(datadir, files)
            print("finishing reading test data")
            test_data_x = []
            test_data_y = []
            for i in test_user_index:  # test
                cur_x = data_all[clients[i]]['x']
                cur_y = data_all[clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(self.tokenizer(cur_x[j]))
                    test_data_y.append(cur_y[j])

            self.data = test_data_x
            self.target = test_data_y
        
        self.build_vocab(vocab_limit_size=20000)
        self.encode(fix_len=300)

    def encode(self, fix_len: int):
        """transform token data to indices sequence by `Vocab`
        Args:
            vocab (fedlab_benchmark.leaf.nlp_utils.util.vocab): vocab for data_token
            fix_len (int): max length of sentence
        Returns:
            list of integer list for data_token, and a list of tensor target
        """
        if len(self.data_tokens_tensor) > 0:
            self.data_tokens_tensor.clear()
            self.targets_tensor.clear()
        self.fix_len = fix_len
        pad_idx = self.vocab.get_index('<pad>')
        print("pad_idx=",pad_idx)
        for tokens in self.data:
            self.data_tokens_tensor.append(
                self.__encode_tokens(tokens, pad_idx))
        for t in self.target:
            self.targets_tensor.append(torch.tensor(t))

    def __encode_tokens(self, tokens, pad_idx) -> torch.Tensor:
        """encode `fix_len` length for token_data to get indices list in `self.vocab`
        if one sentence length is shorter than fix_len, it will use pad word for padding to fix_len
        if one sentence length is longer than fix_len, it will cut the first max_words words
        Args:
            tokens (list[str]): data after tokenizer
        Returns:
            integer list of indices with `fix_len` length for tokens input
        """
        x = [pad_idx for _ in range(self.fix_len)]
        for idx, word in enumerate(tokens[:self.fix_len]):
            x[idx] = self.vocab.get_index(word)
        return torch.tensor(x)

    
    def build_vocab(self, vector_save_root: str = None, vocab_limit_size: int = 50000):
        """Build vocab for dataset with random selected client

        Args:
            vocab_save_root (str): string of path to save built vocab, default to None,
                             which will be modified to "leaf/nlp_utils/dataset_vocab"
            vector_save_root (str): string of path to save pretrain word vector files, default to None,
                             which will be modified to "leaf/nlp_utils/glove"
            vocab_limit_size (int): limit max number of vocab size, default to 50000

        Returns:
            save vocab.pck for dataset
        """
        vector_save_root = './utils/glove'
        print("build the vocab-library for nlp task!")
        self.vocab = Vocab(data_tokens=self.data,
                           word_dim=300,
                           vocab_limit_size=vocab_limit_size,
                           vectors_path=vector_save_root,
                           vector_name='glove.6B.300d.txt')

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, item):
        return self.data_tokens_tensor[item], self.targets_tensor[item]
    
    def get_client_dic(self):
        if self.train:
            return self.dic_users
        else:
            exit("The test dataset do not have dic_users!")

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def split_leaf_data(args):
    # function to split dataset LEAF
    files = os.listdir(args.datadir)
    files = [f for f in files if f.endswith('.json')]
    print(files)
    rng = random.Random(args.seed)

    # check if data contains information on hierarchies
    file_dir = os.path.join(args.datadir, files[0])
    with open(file_dir, 'r') as inf:
        data = json.load(inf)
    include_hierarchy = 'hierarchies' in data
    print("include hierarchy? ", include_hierarchy)

    usrs_samples_num = []
    usrs_num = []
    users = []
    for f in files:
        file_dir = os.path.join(args.datadir, f)
        with open(file_dir, 'r') as inf:
            # Load data into an OrderedDict, to prevent ordering changes
            # and enable reproducibility
            data = json.load(inf, object_pairs_hook=OrderedDict)
            usrs_num.append(len(data['users']))
            users.extend(data['users'])
            usrs_samples_num.extend(data['num_samples'])
    if args.leaf_sample_top:
        indices = sorted(range(len(usrs_samples_num)), key = lambda x:usrs_samples_num[x], reverse=True)
    else:
        indices = list(range(len(usrs_samples_num)))
    num_users = args.leaf_train_num + args.leaf_test_num
    indices = indices[:num_users]
    print("indices: ",indices)
    # print usrs_samples_num with indices as indexes
    print("usrs_samples_num: ",[usrs_samples_num[i] for i in indices])
    
    # print("usrs_samples_num: ",usrs_samples_num)
    rng.shuffle(indices)
    train_indices = indices[:args.leaf_train_num]  # obtain the train index
    test_indices = indices[args.leaf_train_num:]  # obtain the test index

    return train_indices, test_indices, users  # 没有在train_user_index里面的就是test数据

def leaf_read(args):
    train_user_index, test_user_index, users = split_leaf_data(args)
    if args.dataset == 'femnist':
        test_ds = FEMNIST(args.datadir, train_user_index, test_user_index, train=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=32, shuffle=True)
        train_ds = FEMNIST(args.datadir, train_user_index, test_user_index, train=True)
        print(f'number of training / testing samples: {len(train_ds)} / {len(test_ds)}')
    elif args.dataset == 'shakespeare':
        test_ds = ShakeSpeare(args.datadir, train_user_index, test_user_index, train=False)
        test_dl = DataLoader(dataset=test_ds, batch_size=32, shuffle=True)
        train_ds = ShakeSpeare(args.datadir, train_user_index, test_user_index, train=True)
        vocab_size = 80
    train_dataloaders = []
    client_num_samples = []
    dict_users = train_ds.get_client_dic()
    num_classes = args.n_class
    print("num_classes: ", num_classes)
    traindata_cls_counts = []
    for net_id in train_user_index: 
        idxs = dict_users[net_id]
        # print(idxs)
        client_num_samples.append(len(idxs))
        train_ds_local = DatasetSplit(train_ds, idxs)
        train_dl_local = DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_dataloaders.append(train_dl_local)
        traindata_cls_counts.append(count_class(train_ds, num_classes, idxs))
    # print("client_num_samples: ", client_num_samples)
    # 可能数量为0    
    traindata_cls_counts = np.array(traindata_cls_counts)
    data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:,np.newaxis]

    return train_dataloaders, test_dl, client_num_samples, traindata_cls_counts, data_distributions

def sent140_read(args):
    print("-"*40+'sent140'+"-"*40)
    train_user_index, test_user_index, users = split_leaf_data(args)
    test_ds = Sent140(args.datadir, train_user_index, test_user_index, train=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=32, shuffle=True)
    train_ds = Sent140(args.datadir, train_user_index, test_user_index, train=True)
    vocab_size = train_ds.vocab.num
    embedding_dim = train_ds.vocab.word_dim
    pad_idx = train_ds.vocab.get_index('<pad>')
    pretrained_embeddings = train_ds.vocab.vectors
    print("vocab_size=",vocab_size)
    train_dataloaders = []
    client_num_samples = []
    dict_users = train_ds.get_client_dic()
    num_classes = np.unique(train_ds.target).shape[0]
    traindata_cls_counts = []
    for net_id in train_user_index: 
        idxs = dict_users[net_id]
        client_num_samples.append(len(idxs))
        train_ds_local = DatasetSplit(train_ds, idxs)
        train_dl_local = DataLoader(dataset=train_ds_local, batch_size=args.batch_size, shuffle=True, drop_last=True)
        train_dataloaders.append(train_dl_local)
        traindata_cls_counts.append(count_class(train_ds, num_classes, idxs))
    print("client_num_samples: ", client_num_samples)
    traindata_cls_counts = np.array(traindata_cls_counts)
    data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1)[:,np.newaxis]

    return train_dataloaders, test_dl, client_num_samples, traindata_cls_counts, data_distributions, (vocab_size, embedding_dim, pad_idx, pretrained_embeddings)

def count_class(train_ds, num_classes, idxs):
    targets = train_ds.target
    target = [targets[i] for i in idxs]
    traindata_cls_count_per_client = np.zeros(num_classes)
    for t in target:
        traindata_cls_count_per_client[t] += 1
    return traindata_cls_count_per_client