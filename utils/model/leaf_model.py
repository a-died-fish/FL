import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

class CNN_FEMNIST(nn.Module):
    """Used for EMNIST experiments in references[1]
    Args:
        only_digits (bool, optional): If True, uses a final layer with 10 outputs, for use with the
            digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
            If selfalse, uses 62 outputs for selfederated Extended MNIST (selfEMNIST)
            EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373
            Defaluts to `True`
    Returns:
        A `torch.nn.Module`.
    """
    def __init__(self):
        super(CNN_FEMNIST, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        # x = self.softmax(x)
        x = self.fc(x)
        return x
    
class RNN_Shakespeare(nn.Module):
    def __init__(self, vocab_size=80, embedding_dim=8, hidden_size=256):
        """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
        Args:
            vocab_size (int, optional): the size of the vocabulary, used as a dimension in the input embedding,
                Defaults to 80.
            embedding_dim (int, optional): the size of embedding vector size, used as a dimension in the output embedding,
                Defaults to 8.
            hidden_size (int, optional): the size of hidden layer. Defaults to 256.
        Returns:
            A `torch.nn.Module`.
        Examples:
            RNN_Shakespeare(
              (embeddings): Embedding(80, 8, padding_idx=0)
              (lstm): LSTM(8, 256, num_layers=2, batch_first=True)
              (fc): Linear(in_features=256, out_features=90, bias=True)
            ), total 819920 parameters
        """
        super(RNN_Shakespeare, self).__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True)
        # self.fc = nn.Linear(hidden_size, vocab_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeds)
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output
    
class LSTMModel(nn.Module):
    def __init__(self,
                 vocab_size, embedding_dim, hidden_size=64, num_layers=2, output_dim=2, pad_idx=0,
                 using_pretrained=False, embedding_weights=None, bid=False):
        """Creates a RNN model using LSTM layers providing embedding_weights to pretrain
        Args:
            vocab_size (int): the size of the vocabulary, used as a dimension in the input embedding
            embedding_dim (int): the size of embedding vector size, used as a dimension in the output embedding
            hidden_size (int): the size of hidden layer, e.g. `256`
            num_layers (int): the number of recurrent layers, e.g. `2`
            output_dim (int): the dimension of output, e.g. `10`
            pad_idx (int): the index of pad_token
            using_pretrained (bool, optional): if use embedding vector to pretrain model, set `True`, defaults to `False`
            embedding_weights (torch.Tensor, optional): vectors to pretrain model, defaults to `None`
            bid (bool, optional): if use bidirectional LSTM model, set `True`, defaults to `False`
        Returns:
            A `torch.nn.Module`.
        """
        super(LSTMModel, self).__init__()
        self.bid = bid
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=pad_idx)
        if using_pretrained:
            assert embedding_weights.shape[0] == vocab_size
            assert embedding_weights.shape[1] == embedding_dim
            self.embeddings.from_pretrained(embedding_weights)
            # self.embedding.weight.data.copy_(embedding_weights)

        self.dropout = nn.Dropout(0.5)
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bid,
            dropout=0.5,
            batch_first=True
        )

        # using bidrectional, *2
        if bid:
            hidden_size *= 2
        self.fc = nn.Linear(hidden_size, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_seq: torch.Tensor):
        embeds = self.embeddings(input_seq)  # (batch, seq_len, embedding_dim) [64,300,300]
        embeds = self.dropout(embeds)
        lstm_out, (hidden, cell) = self.encoder(embeds)
        # outputs [seq_len, batch, hidden*2] *2 means using bidrectional
        if self.bid:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        
        output = self.fc(hidden)
        # print("before: output=",output[0][:])
        output = self.sigmoid(output)
        # print("after: output=",output[0][:])
        return output