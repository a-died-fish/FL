import torch
import copy
import random
import os

from config import get_args
from utils import *
from transmit import create_ssh_client,ensure_remote_path_exists_and_writable,transfer_file,check_file_exists

args = get_args()
set_seed(args.seed)
if args.dataset == 'sent140':
    train_dls, test_dl, client_num_samples, traindata_cls_counts, data_distributions, sent140_info = get_dataloader(args)
elif args.dataset =='pacs' or args.dataset == 'officehome' or args.dataset == 'vlcs':
    train_dls, test_dl, client_num_samples, domain = get_dataloader(args) # test_dl is a list consisting 4 dataloader here
    sent140_info = None
else:
    train_dls, test_dl, client_num_samples, traindata_cls_counts, data_distributions = get_dataloader(args)
    sent140_info=None

init_model_dir = '/GPFS/data/xinyuzhu-1/FL/models/init_model'
global_model_dir = '/GPFS/data/xinyuzhu-1/FL/models/global_models'
local_model_dir = '/GPFS/data/xinyuzhu-1/FL/models/local_models'
global_model = get_model(args, sent140_info)
model_state = global_model.state_dict()
# torch.save(model_state, os.path.join(init_model_dir,'init_model.pth'))
# global_model.load_state_dict(torch.load(args.load_path)) if args.load_path is not None else None    # Load saved model if exists
global_model.load_state_dict(torch.load(os.path.join(init_model_dir,'init_model')))
local_model_list = [copy.deepcopy(global_model) for _ in range(args.n_client)]
local_extra_list = [set_model_zero(copy.deepcopy(global_model)) for _ in range(args.n_client)] if args.alg in ('scaffold') else []
global_extra = set_model_zero(copy.deepcopy(global_model)) if args.alg in ('scaffold') else []

best_acc = 0.0


server = "60.204.226.214"
port = 22  
user = "root"
password = "Zhuxinyu13579"
remote_path = "/data/FL/local_models/"  

# ========== Federated Learning ==========
print(f"{'='*20} Start Federated Learning {'='*20}")  
for round in range(args.n_round):

    # Choose available clients
    # available_clients = sorted(random.sample(range(args.n_client), int(args.sample_fraction * args.n_client)))
    available_clients = [0] # change this to your user id

    if args.sample_fraction < 1.0:
        print('>> Round {} | Available Clients: {} <<'.format(round, available_clients))
    # Federated Process: sync model / local train / global aggregate / global evaluate
    if args.alg == 'fedavg':
        local_train_base(args, train_dls, local_model_list, global_model, available_clients)

        # save local model
        model_state = local_model_list[available_clients[0]].state_dict()
        torch.save(model_state,os.path.join(local_model_dir,'client_'+str(available_clients[0])+'_round_'+str(round)+'.pth'))
        local_file = os.path.join(local_model_dir,'client_'+str(available_clients[0])+'_round_'+str(round)+'.pth')

        # upload the model to server
        transfer_file(local_file, remote_path, server, port, user, password)

        # check whether the server sends the global model
        check_file_exists(global_model_dir,'round_'+str(round)+'.pth')

        # load the global model send by server
        global_model.load_state_dict(torch.load(os.path.join(global_model_dir,'round_'+str(round)+'.pth')))

        # no need to aggregate again
        # aggregate_model(global_model, local_model_list, client_num_samples, available_clients)
        acc, best_acc = evaluation(args, global_model, test_dl, best_acc)

    elif args.alg == 'fedprox':
        local_train_fedprox(args, train_dls, local_model_list, global_model, available_clients)
        aggregate_model(global_model, local_model_list, client_num_samples, available_clients)
        acc, best_acc = evaluation(args, global_model, test_dl, best_acc)
    elif args.alg == 'scaffold':
        local_train_scaffold(args, train_dls, local_model_list, global_model, available_clients, local_extra_list, global_extra)
        aggregate_model(global_model, local_model_list, client_num_samples, available_clients)
        acc, best_acc = evaluation(args, global_model, test_dl, best_acc)

    print('>> Round {} | Current Acc: {:.5f}, Best Acc: {:.5f} <<\n{}'.format(round, acc, best_acc, '-'*80))