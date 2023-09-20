from numpy import random
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import sys
import collections
import pdb

def index_from_set(x, allowable_set):
    length = len(allowable_set)
    if x not in allowable_set:
        return length - 1
    return allowable_set.index(x)

def remove_duplicates_with_b(input_list_a, input_list_b):
    seen = {}
    result_a = []
    result_b = []
    
    for a, b in zip(input_list_a, input_list_b):
        a_tuple = tuple(sorted(a))
        if a_tuple not in seen:
            seen[a_tuple] = True
            result_a.append(a)
            result_b.append(b)
    
    return result_a, result_b

def idx2one_hot(i, allow_list):
    result = [0] * len(allow_list)
    if i in allow_list:
        result[allow_list.index(i)] = 1
    else:
        result[-1] = 1
    return result

def num2one_hot(i, max_num):
    result = [0] * max_num
    result[i] = 1
    return result

'''
手性、立体结构等不能通过直接构造取得，只能根据距离来算出
分子的一个原子必要的特征有：
    原子序数：  i.GetAtomicNum()
    质量：      i.GetMass()
    H的数量     i.GetNumExplicitHs()
    形式电荷    i.GetFormalCharge()
    度数        i.GetDegree()
    显式价数    i.GetExplicitValence()
    隐式价数    i.GetImplicitValence()
    杂化方式    i.GetHybridization()
    是否为芳香环i.GetIsAromatic()
分子的一个键的必要特征：
    键类型      i.GetBondType()
    共轭        i.GetIsConjugated()
    是否为环    i.IsInRing() 
'''


@dataclass
class AtomConfig:
    one_hot: bool
    element_type: bool      #元素类型
    mass:bool               #质量
    num_of_H:bool           #H的数量
    formal_charge: bool     #形式电荷
    degree: bool            #度数
    explicit_valence: bool  #显式价数
    implicit_valence: bool  #隐式价数
    hybridization: bool     #杂化方式
    aromatic: bool          #芳香性


element_names = ['*', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br','I',"unk"]
hybridization_type = [
                Chem.rdchem.HybridizationType.S,
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                # Chem.rdchem.HybridizationType.SP2D,
                # Chem.rdchem.HybridizationType.SP3D,
                # Chem.rdchem.HybridizationType.SP3D2,
                Chem.rdchem.HybridizationType.OTHER
            ]


@dataclass
class BondConfig:
    one_hot: bool
    bond_type: bool     #键类型
    conjugation: bool   #共轭
    ring: bool          #是否为环

bond_type = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
            # Chem.rdchem.BondType.QUADRUPLE,
            # Chem.rdchem.BondType.QUINTUPLE ,
            # Chem.rdchem.BondType.HEXTUPLE ,
            # Chem.rdchem.BondType.OTHER,
            "unk"
        ]

def atom_fp(atom, atom_config: AtomConfig):
    results = []
    if atom_config.one_hot:
        if atom_config.element_type:
            results.extend(idx2one_hot(atom.GetSymbol(),element_names))

        # if atom_config.mass:
        #     results.append(atom.GetMass()/12)

        # if atom_config.num_of_H:
        #     results.append(atom.GetNumExplicitHs())

        if atom_config.formal_charge:
            results.extend(num2one_hot(atom.GetFormalCharge()-1,3))   # 统计出来最大为3

        if atom_config.degree:
            results.extend(num2one_hot(atom.GetDegree()-1,4))         # 统计出来最大为4

        if atom_config.explicit_valence:
            results.extend(num2one_hot(atom.GetExplicitValence()-1,7))    # 抽样出来最大为6

        # if atom_config.implicit_valence:
        #     results.append(atom.GetImplicitValence())

        if atom_config.hybridization:
            results.extend(idx2one_hot(atom.GetHybridization(), hybridization_type))    # 抽样出来为4个

        if atom_config.aromatic:
            results.append(1 if atom.GetIsAromatic() else 0)
    else:   #有待修改
        if atom_config.element_type:
            results.append(index_from_set(atom.GetSymbol(),element_names))

        if atom_config.mass:
            results.append(atom.GetMass()/12)

        if atom_config.num_of_H:
            results.append(atom.GetNumExplicitHs())

        if atom_config.formal_charge:
            results.append(atom.GetFormalCharge())

        if atom_config.degree:
            results.append(atom.GetDegree())

        if atom_config.explicit_valence:
            results.append(atom.GetExplicitValence())

        if atom_config.implicit_valence:
            results.append(atom.GetImplicitValence())

        if atom_config.hybridization:
            results.append(index_from_set(atom.GetHybridization(), hybridization_type))

        if atom_config.aromatic:
            results.append(1 if atom.GetIsAromatic() else 0)
    # print(results)
    return results

def atom_helper(molecule, ind, atom_config):
    atom = molecule.GetAtomWithIdx(ind)
    atom_feature = atom_fp(atom, atom_config)
    return atom_feature

def bond_fp(bond, config):
    bond_feats = []

    if config.one_hot:
        if config.bond_type:
            bond_feats.extend(idx2one_hot(bond.GetBondType(),bond_type))

        if config.conjugation:
            bond_feats.append(1 if bond.GetIsConjugated() else 0)

        if config.ring:
            bond_feats.append(1 if bond.IsInRing() else 0)
    else:   #有待修改
        if config.bond_type:
            bond_feats.append(index_from_set(bond.GetBondType(),bond_type))

        if config.conjugation:
            bond_feats.append(1 if bond.GetIsConjugated() else 0)

        if config.ring:
            bond_feats.append(1 if bond.IsInRing() else 0)

    return bond_feats

bond_config = BondConfig(True, True, True, True)
atom_config = AtomConfig(
    True,
    True,   # element_type
    False,   # mass
    False,   # num_of_H
    True,   # formal_charge
    True,   # degree
    True,   # explicit_valence
    False,   # implicit_valence
    True,   # hybridization
    True    # aromatic
)



def get_feature(smiles):
    smiles = Chem.CanonSmiles(smiles)
    molecule = Chem.MolFromSmiles(smiles)
    n_atoms = molecule.GetNumAtoms()

    node_features = [atom_helper(molecule, i, atom_config) for i in range(0, n_atoms)]
    edge_features = []
    edge_indices = []
    for bond in molecule.GetBonds():
        edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_indices.append(
            [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        )
        bond_feature = bond_fp(bond, bond_config)
        edge_features.extend(
            [bond_feature, bond_feature]  # both bonds have the same features
        )


    return node_features, edge_features, edge_indices
