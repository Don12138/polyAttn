import sklearn
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Union
import numpy as np
from itertools import zip_longest
from copy import deepcopy
from collections import Counter
from torch_scatter import scatter_sum, scatter_mean
import logging
from rdkit import Chem
import math
from onmt.modules.embeddings import PositionalEncoding
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.utils.misc import sequence_mask
import sys
sys.path.append("/home/chenlidong/polyAttn/models")
sys.path.append("/home/chenlidong/polyAttn/utils")
# import polygnn
sys.path.append("/home/chenlidong/polymer-chemprop-master/chemprop/")
import chem_utils
import pdb
from torch.nn.utils.rnn import pad_sequence
from typing import List, Union, Tuple
from functools import reduce
from rdkit import Chem
import torch.nn.init as init
import networkx as nx
from args import TrainArgs
from features import mol2graph, BatchMolGraph, get_atom_fdim, get_bond_fdim
from nn_utils import get_activation_function, index_select_ND