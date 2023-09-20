import pandas as pd
import numpy as np
import sys
sys.path.append("/home/chenlidong/polymer-chemprop-master/")
from chemprop.features import mol2graph
import os
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, download_url, Data


class PolymerDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, phase="train", atom_messages=False):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.phase = phase
        self.atom_messages = atom_messages
        self.dataset = []
        self.tensor2data()

    @property
    def raw_file_names(self):
        if self.phase == "train":
            return [os.path.join(self.root, "train.pt")]
        elif self.phase == "val":
            return [os.path.join(self.root, "val.pt")]
        elif self.phase == "test":
            return [os.path.join(self.root, "test.pt")]
        else:
            raise ValueError("phase must be in [train, val, test]")

    @property
    def processed_file_names(self):
        return self.dataset
        
    def tensor2data(self):
        # Todo
        # add atom_message information
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            f_atoms, f_bonds, w_atoms, w_bonds, a2b, b2a,\
            b2revb, a_scope, b_scope, a2a, b2b, ea, ip = torch.load(raw_path)

            for idx,(a_scope_i,b_scope_i) in enumerate(zip(a_scope,b_scope)):
                edge_index = [[],[]]
                b2a_i = b2a[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]]
                b2revb_i = b2revb[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]]

                for i in range(b2a_i.size(0)):
                    edge_index[0].append(b2a_i[i].item()-a_scope_i[0])
                    edge_index[1].append(b2a_i[b2revb_i[i]-b_scope_i[0]]-a_scope_i[0])

                data = Data(
                    x = f_atoms[a_scope_i[0]:a_scope_i[0]+a_scope_i[1]],
                    edge_index=torch.tensor(edge_index,dtype=torch.long),
                    edge_attr = f_bonds[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]],
                    w_atoms = w_atoms[a_scope_i[0]:a_scope_i[0]+a_scope_i[1]],
                    w_bonds = w_bonds[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]],
                    # a2b = a2b[a_scope_i[0]:a_scope_i[0]+a_scope_i[1]],
                    # b2a = b2a[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]],
                    # b2revb = b2revb[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]],
                    # a2a = a2a[a_scope_i[0]:a_scope_i[0]+a_scope_i[1]],
                    # b2b = b2b[b_scope_i[0]:b_scope_i[0]+b_scope_i[1]],
                    ea = ea[idx],
                    ip = ip[idx]
                )
                self.dataset.append(data)
            

    def get(self,idx):
        return self.dataset[idx]
    
    def len(self):
        return len(self.dataset)