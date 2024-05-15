from .resnet_cifar import *
from .resnet_cifar import ModelFedCon_noheader
from .leaf_model import *

Name2Function = {
    'resnet20_cifar': resnet20_cifar,
    'resnet32_cifar': resnet32_cifar,
    'resnet44_cifar': resnet44_cifar,
    'resnet56_cifar': resnet56_cifar,
    'resnet110_cifar': resnet110_cifar,
    'resnet1202_cifar': resnet1202_cifar
}

def get_model(args, sent140_info=None):
    if args.model in Name2Function:
        return Name2Function[args.model](args.n_class)
    elif args.model == 'lstm':
        vocab_size, embedding_dim, pad_idx, pretrained_embeddings = sent140_info
        return LSTMModel(vocab_size, embedding_dim, output_dim=args.n_class, pad_idx=pad_idx,
                 using_pretrained=True, embedding_weights=pretrained_embeddings, bid=True)
    elif args.model == 'lstm_shakes':
        return RNN_Shakespeare()
    elif args.model == 'cnn_femnist':
        return CNN_FEMNIST()
    elif args.model == 'resnet18_7':
         return ModelFedCon_noheader('resnet18_7',args.n_class)
    elif args.model == 'resnet18_7_gn':
         return ModelFedCon_noheader('resnet18_7_gn',args.n_class)
    elif args.model == 'resnet50_7':
         return ModelFedCon_noheader('resnet50_7',args.n_class)
    # return eval(args.model)(args.n_class)     # this also works if the args.model is exactly the name of a function