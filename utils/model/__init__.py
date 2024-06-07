from .resnet_cifar import *


Name2Function = {
    'resnet20_cifar': resnet20_cifar,
    'resnet32_cifar': resnet32_cifar,
    'resnet44_cifar': resnet44_cifar,
    'resnet56_cifar': resnet56_cifar,
    'resnet110_cifar': resnet110_cifar,
    'resnet1202_cifar': resnet1202_cifar
}

def get_model(args):
    if args.model in Name2Function:
        return Name2Function[args.model](args.n_class)