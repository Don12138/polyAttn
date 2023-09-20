import torch
import torch.nn as nn
import torch_geometric.nn as pnn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import pdb
import math


def get_unit_sequence(input_dim, output_dim, n_hidden):
    """
    Smoothly decay the number of hidden units in each layer.
    Start from 'input_dim' and end with 'output_dim'.

    Examples:
    get_unit_sequence(1,1024,4) = [1, 4, 16, 64, 256, 1024]
    get_unit_sequence(1024,1,4) = [1024, 256, 64, 16, 4, 1]
    """
    reverse = False
    if input_dim > output_dim:
        reverse = True
        input_dim,output_dim = output_dim,input_dim

    diff = abs(output_dim.bit_length() - input_dim.bit_length())
    increment = diff // (n_hidden+1)

    sequence = [input_dim] + [0] * (n_hidden) + [output_dim]

    for idx in range(n_hidden // 2):
        sequence[idx+1] = 2 ** ((sequence[idx]).bit_length() + increment-1)
        sequence[-2-idx] = 2 ** ((sequence[-1-idx]-1).bit_length() - increment)

    if n_hidden%2 == 1:
        sequence[n_hidden // 2 + 1] = (sequence[n_hidden // 2] + sequence[n_hidden // 2+2])//2

    if reverse: 
        sequence.reverse()

    return sequence

class output(nn.Module):
    """
    Output layer with xavier initialization on weights
    Output layer with target mean (plus noise) on bias. Suggestion from: http://karpathy.github.io/2019/04/25/recipe/
    """

    def __init__(self, size_in, target_mean=[0]):
        super().__init__()
        self.size_in, self.size_out = size_in, len(target_mean)
        self.target_mean = target_mean

        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.target_mean != None:
            self.linear.bias.data = torch.tensor(target_mean)

    def forward(self, x):
        return self.linear(x)

class hidden_xavier_bn(nn.Module):
    """
    Hidden layer with xavier initialization and batch normalization
    """
    def __init__(self, size_in, size_out, hps):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        self.linear = nn.Linear(self.size_in, self.size_out)
        nn.init.xavier_uniform_(self.linear.weight)
        self.bn = nn.BatchNorm1d(self.size_out)
        self.activation = hps['activation']

    def forward(self, x):
        # print(x)
        return self.activation(self.bn(self.linear(x)))

class ffn(nn.Module):
    """
    A Feed-Forward neural Network that uses DenseHidden layers
    """

    def __init__(self, input_dim, output_dim, hps):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hps = hps
        self.layers = nn.ModuleList()
        self.unit_sequence = get_unit_sequence(
            input_dim, output_dim, self.hps['ffn_capacity']
        )
        # set up hidden layers
        for ind, n_units in enumerate(self.unit_sequence[:-1]):
            size_out_ = self.unit_sequence[ind + 1]
            self.layers.append(
                hidden_xavier_bn(
                    size_in=n_units,
                    size_out=size_out_,
                    hps=self.hps,
                )
            )

    def forward(self, x):
        """
        Compute the forward pass of this model
        """
        for layer in self.layers:
            x = layer(x)

        return x