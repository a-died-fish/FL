import torch
import os
from transmit import create_ssh_client,ensure_remote_path_exists_and_writable,transfer_file,check_file_exists

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
import torchvision.models as models
import copy
import time

client_num = 3
local_model_dir = '/data/FL/local_models'
global_model_dir = '/data/FL/global_models'

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetBase(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetBase, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.base = ResNetBase(block, num_blocks)
        self.classifier = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def forward(self, x):
        out = self.base(x)
        out = self.classifier(out)
        return out
    
def resnet20_cifar(num_classes):
    return ResNet(BasicBlock, [3, 3, 3], num_classes)

Name2Function = {
    'resnet20_cifar': resnet20_cifar,
}

def get_model(model_name,n_class):
    if model_name in Name2Function:
        return Name2Function[model_name](n_class)
    



def check_file_exists(directory, filename):
    filepath = os.path.join(directory, filename)
    while not os.path.exists(filepath):
        print(f"文件 {filename} 尚未出现，正在等待...")
        time.sleep(5)  # 等待5秒后再次检查
    print(f"文件 {filename} 已出现。")

def aggregate_model(global_model, local_model_list, client_num_samples, available_clients):
    total_data_points = sum([client_num_samples[r] for r in available_clients])
    fed_avg_freqs = [client_num_samples[r] / total_data_points for r in available_clients]
    
    global_w = global_model.state_dict()
    for net_id, client_id in enumerate(available_clients):
        net_para = local_model_list[client_id].state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]
    global_model.load_state_dict(global_w)


global_model = get_model('resnet20_cifar',10)
local_model_list = [copy.deepcopy(global_model) for _ in range(client_num)]
round=0
available_clients = [0,1,2]
client_num_samples = [18177,13635,18188]

# client 0
server_0 = "1.95.12.201"
port = 22  
user = "root"
password_0 = ""
remote_path_0 = "/data/FL/models/global_models/" 

# client 1
server_1 = "124.71.190.67"
port = 22  
user = "root"
password_1 = ""
remote_path_1 = "/root/data/FL/models/global_models"  

# client 2
server_2 = "124.71.212.18"
port = 22  
user = "root"
password_2 = ""
remote_path_2 = "/root/data/FL/models/global_models"  

while True:
    for client_id in range(client_num):
        check_file_exists(local_model_dir,'client_'+str(client_id)+'_round_'+str(round)+'.pth')
        time.sleep(5)
        local_model_list[client_id].load_state_dict(torch.load(os.path.join(local_model_dir,'client_'+str(client_id)+'_round_'+str(round)+'.pth')))

    aggregate_model(global_model, local_model_list, client_num_samples, available_clients)
    global_model_state = global_model.state_dict()
    torch.save(global_model_state, os.path.join(global_model_dir,'round_'+str(round)+'.pth'))
    transfer_file(os.path.join(global_model_dir,'round_'+str(round)+'.pth'), remote_path_0, server_0, port, user, password_0)
    time.sleep(2)
    transfer_file(os.path.join(global_model_dir,'round_'+str(round)+'.pth'), remote_path_1, server_1, port, user, password_1)
    time.sleep(2)
    transfer_file(os.path.join(global_model_dir,'round_'+str(round)+'.pth'), remote_path_2, server_2, port, user, password_2)
    time.sleep(2)
    round = round +1




