import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Union
import numpy as np
from itertools import zip_longest
from copy import deepcopy
from collections import Counter
import logging
from rdkit import Chem

import sys
sys.path.append("/home/chenlidong/polyAttn/models")
sys.path.append("/home/chenlidong/polyAttn/utils")
# import polygnn
import chem_utils
import pdb
import sys
sys.path.append("/home/chenlidong/polymer-chemprop-master/chemprop")
from features import mol2graph
from typing import List, Union, Tuple
from functools import reduce
from rdkit import Chem
from args import TrainArgs
from features import BatchMolGraph, get_atom_fdim, get_bond_fdim
from nn_utils import get_activation_function, index_select_ND
from data_utils import PolymerDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric import nn as pnn
a = mol2graph(['[*:1]c1cc(F)c([*:2])cc1F.[*:3]c1cc(F)c([*:4])cc1C#N|0.5|0.5|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125'])
torch.manual_seed(42)

class MPNEncoder(nn.Module):
    """An :class:`MPNEncoder` is a message passing neural network for encoding a molecule."""

    def __init__(self, args: TrainArgs, atom_fdim: int, bond_fdim: int):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        :param atom_fdim: Atom feature vector dimension.
        :param bond_fdim: Bond feature vector dimension.
        """
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim                  # atom feature len
        self.bond_fdim = bond_fdim                  # bond feature len
        self.atom_messages = args.atom_messages     # 是否以原子为中心传递
        self.hidden_size = args.hidden_size         # 300
        self.bias = args.bias
        self.depth = args.depth                     # 消息传递的步数
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected           # False
        self.device = torch.device('cpu')
        self.aggregation = args.aggregation         # mean
        self.aggregation_norm = args.aggregation_norm   # 100

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation_function(args.activation)    # Relu

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # layer after concatenating the descriptors if args.atom_descriptors == descriptors
        if args.atom_descriptors == 'descriptor':
            self.atom_descriptors_size = args.atom_descriptors_size
            self.atom_descriptors_layer = nn.Linear(self.hidden_size + self.atom_descriptors_size,
                                                    self.hidden_size + self.atom_descriptors_size,)

    def forward(self,
                x,
                atom_descriptors_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A :class:`~chemprop.features.featurization.BatchMolGraph` representing
                          a batch of molecular graphs.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atomic descriptors
        :return: A PyTorch tensor of shape :code:`(num_molecules, hidden_size)` containing the encoding of each molecule.
        """
        if atom_descriptors_batch is not None:
            atom_descriptors_batch = [np.zeros([1, atom_descriptors_batch[0].shape[1]])] + atom_descriptors_batch   # padding the first with 0 to match the atom_hiddens
            atom_descriptors_batch = torch.from_numpy(np.concatenate(atom_descriptors_batch, axis=0)).float().to(self.device)

        f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb, \
        a_scope, b_scope, degree_of_polym = x.get_components(atom_messages=self.atom_messages)

        # f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a, b2revb, a2a, b2b = x.f_atoms.to(self.device), x.f_bonds.to(self.device), \
        #                                                        x.w_atoms.to(self.device), x.w_bonds.to(self.device), \
        #                                                        x.a2b.to(self.device), x.b2a.to(self.device), \
        #                                                        x.b2revb.to(self.device), x.a2a.to(self.device), x.b2b.to(self.device)
        

        pdb.set_trace()

        if self.atom_messages:
            a2a = mol_graph.get_a2a().to(self.device)


        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)                                   # [num_atoms , hidden_size]
        else:
            input = self.W_i(f_bonds)                                   # [num_bonds , hidden_size]
        message = self.act_func(input)                                  # [num_bonds / num_atoms , hidden_size]


        pdb.set_trace()
        # Message passing                                               
        for depth in range(self.depth - 1):
            if self.undirected:
                message = (message + message[b2revb]) / 2

            if self.atom_messages:
                nei_a_message = index_select_ND(message, a2a)           # [num_atoms , max_num_bonds , hidden_size] 某一原子，其周围所有原子的特征
                nei_f_bonds = index_select_ND(f_bonds, a2b)             # [num_atoms , max_num_bonds , bond_fdim]
                nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # [num_atoms , max_num_bonds , hidden + bond_fdim]
                message = nei_message.sum(dim=1)                        # [num_atoms , hidden + bond_fdim]
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)           # [num_atoms , max_num_bonds , hidden]
                nei_a_weight = index_select_ND(w_bonds, a2b)            # [num_atoms , max_num_bonds]
                # weight nei_a_message based on edge weights
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1) * weight(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = dot(nei_a_message,nei_a_weight)      rev_message
                nei_a_message = nei_a_message * nei_a_weight[..., None]  # [num_atoms , max_num_bonds , hidden]
                a_message = nei_a_message.sum(dim=1)                     # [num_atoms , hidden]
                rev_message = message[b2revb]                            # [num_bonds , hidden]
                message = a_message[b2a] - rev_message                   # [num_bonds , hidden]

            message = self.W_h(message)
            message = self.act_func(input + message)                    # [num_bonds/num_atoms , hidden]  skip connection
            message = self.dropout_layer(message)                       # [num_bonds/num_atoms , hidden]


        pdb.set_trace()

        a2x = a2a if self.atom_messages else a2b
        nei_a_message = index_select_ND(message, a2x)                   # [num_atoms , max_num_bonds , hidden]
        nei_a_weight = index_select_ND(w_bonds, a2x)                    # [num_atoms , max_num_bonds]
        # weight messages
        nei_a_message = nei_a_message * nei_a_weight[..., None]         # [num_atoms , max_num_bonds , hidden]
        a_message = nei_a_message.sum(dim=1)                            # [num_atoms , hidden]
        a_input = torch.cat([f_atoms, a_message], dim=1)                # [num_atoms , hidden + f_atom]
        atom_hiddens = self.act_func(self.W_o(a_input))                 # [num_atoms , hidden]
        atom_hiddens = self.dropout_layer(atom_hiddens)                 # [num_atoms , hidden]

        # concatenate the atom descriptors
        if atom_descriptors_batch is not None:
            if len(atom_hiddens) != len(atom_descriptors_batch):
                raise ValueError(f'The number of atoms is different from the length of the extra atom features')

            atom_hiddens = torch.cat([atom_hiddens, atom_descriptors_batch], dim=1)     # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.atom_descriptors_layer(atom_hiddens)                    # num_atoms x (hidden + descriptor size)
            atom_hiddens = self.dropout_layer(atom_hiddens)                             # num_atoms x (hidden + descriptor size)

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)                   # 得到第i个聚合物的张量
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)                       
                w_atom_vec = w_atoms.narrow(0, a_start, a_size)                         # 得到第i个聚合物单体的比例
                # if input are polymers, weight atoms from each repeating unit according to specified monomer fractions
                # weight h by atom weights (weights are all 1 for non-polymer input)
                mol_vec = w_atom_vec[..., None] * mol_vec
                # weight each atoms at readout
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / w_atom_vec.sum(dim=0)  # if not --polymer, w_atom_vec.sum == a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                elif self.aggregation == 'norm':
                    mol_vec = mol_vec.sum(dim=0) / self.aggregation_norm

                # if input are polymers, multiply mol vectors by degree of polymerization
                # if not --polymer, Xn is 1
                mol_vec = degree_of_polym[i] * mol_vec

                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class poly_chemprop(pnn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')  # "Add" aggregation (Step 5).
        self.atom_fdim = atom_fdim                  # atom feature len
        self.bond_fdim = bond_fdim                  # bond feature len
        self.atom_messages = args.atom_messages     # 是否以原子为中心传递
        self.hidden_size = args.hidden_size         # 300
        self.bias = args.bias
        self.depth = args.depth                     # 消息传递的步数
        self.dropout = args.dropout
        self.undirected = args.undirected           # False
        self.device = args.device

        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.act_func = get_activation_function(args.activation)
        input_dim = self.atom_fdim if self.atom_messages else self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        if self.atom_messages:
            w_h_input_size = self.hidden_size + self.bond_fdim
        else:
            w_h_input_size = self.hidden_size

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(w_h_input_size, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

    # def forward(self,batch)

model = MPNEncoder(args=TrainArgs, atom_fdim=get_atom_fdim(),bond_fdim=get_bond_fdim(atom_messages=False))

# train_dataset = PolymerDataset("/home/chenlidong/polyAttn/preprocessed/copolymer_4w",phase="train")
# val_dataset = PolymerDataset("/home/chenlidong/polyAttn/preprocessed/copolymer_4w",phase="val")
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
# for batch in train_loader:
#     print(model(batch))
#     break
model(a)