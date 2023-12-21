import sys
sys.path.append("/home/chenlidong/polymer-chemprop-master/chemprop/")
from args import TrainArgs
from features import mol2graph, BatchMolGraph, get_atom_fdim, get_bond_fdim
from nn_utils import get_activation_function, index_select_ND
from models import MoleculeModel
from data import MoleculeDatapoint,MoleculeDataset
import pdb
args = TrainArgs()
args.data_path ="/home/chenlidong/polymer-chemprop-master/data/polymer-chemprop-data-main/results/vipea/cv9-monomer/master/dataset-master_chemprop.csv",
args.dataset_type = "regression"
args.aggregation = "mean"
args.polymer = False
args.with_pe = True
args.checkpoint_frzn = "/home/chenlidong/polymer-chemprop-master/data/polymer-chemprop-data-main/results/vipea/cv9-monomer/master/chemprop_checkpoints_wDMPNN_pe/fold_0/model_0/model.pt"
# args.process_args()
args.ffn_hidden_size = args.hidden_size

model = MoleculeModel(args)
model.to("cuda")

point = MoleculeDatapoint(smiles=["c1ccc2c(c1)[nH]c1ccccc12.Oc1ccc2ccccc2c1-c1c(O)ccc2ccccc12"],
                 targets= [1.,2.,3.],
                 idx=8864,
                 mtype=0,
                 data_weight= 1,
                #  features: np.ndarray = None,
                #  features_generator: List[str] = None,
                #  phase_features: List[float] = None,
                #  atom_features: np.ndarray = None,
                #  atom_descriptors: np.ndarray = None,
                #  bond_features: np.ndarray = None,
                #  overwrite_default_atom_features: bool = False,
                #  overwrite_default_bond_features: bool = False
                 )


batch = MoleculeDataset([point])

mol_batch, features_batch, target_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights()


print(model(mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch))