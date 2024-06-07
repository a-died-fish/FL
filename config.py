import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_model', action="store_true", help="whether to save the global model.")
    parser.add_argument('--local_eval', action="store_true", help="whether to evaluate local model.")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--datadir', type=str, default="./data", help="Data directory")

    # Training
    parser.add_argument('--model', type=str, default='resnet20_cifar', help='neural network used in training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--n_iteration', type=int, default=100, help='number of local iterations')

    # FL Setting
    parser.add_argument('--alg', type=str, default='fedavg', help='federated learning algorithm to run')
    parser.add_argument('--n_client', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--n_round', type=int, default=100, help='number of maximum communication roun')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default=2, help='The parameter for the noniid-skew for data partitioning')

    # FedProx / MOON
    parser.add_argument('--mu', type=float, default=0.01)

    
    args = parser.parse_args()
    
    dataset_info = {
        'cifar10': {'n_class': 10, 'n_channel': 3, 'img_size': '32'},
        'eurosat':{'n_class': 10,'n_channel': 3, 'img_size': '64'}
    }

    args.n_class = dataset_info[args.dataset]['n_class']
    args.n_channel = dataset_info[args.dataset]['n_channel']
    args.img_size = dataset_info[args.dataset]['img_size']
    

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print_parsed_args(args)

    return args

def print_parsed_args(args):
    print(f"{'='*20} Parsed Arguments {'='*20}")  
  
    # Determine the maximum length of argument names  
    max_key_length = max(len(key) for key in vars(args).keys())  
  
    # Iterate through the parsed arguments and print them with proper padding  
    for key, value in vars(args).items():  
        print(f"{key.ljust(max_key_length)}: {value}")  
  
    print('=' * 40)