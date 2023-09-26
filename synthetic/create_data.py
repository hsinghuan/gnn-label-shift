import numpy as np
import os
import torch
from torch_geometric.datasets.sbm_dataset import StochasticBlockModelDataset

from dataset import StochasticBlockModelBlobDataset
import sys
sys.path.append("..")
from utils import set_model_seed

SEED = 42

def create_sbm_ls_low_dim(data_dir, src_blk_sz, tgt_blk_sz, src_edge_probs, tgt_edge_probs, centers, flip_y=0.,
                          src_split=[0.8,0.2], tgt_split=[0.6,0.2]):
    """
    Creates a source and a target StochasticBlockModelBlobDataset, which has low dimensional node features where the centers are user-specified
    :param data_dir:
    :param src_blk_sz:
    :param tgt_blk_sz:
    :param src_edge_probs:
    :param tgt_edge_probs:
    :param centers:
    :param flip_y:
    :return:
    """
    class_num = centers.shape[0]
    feat_dim = centers.shape[1]
    assert src_edge_probs.shape == tgt_edge_probs.shape
    assert src_edge_probs.shape[0] == class_num
    os.makedirs(os.path.join(data_dir, "src"), exist_ok=True)
    src_dataset = StochasticBlockModelBlobDataset(root=os.path.join(data_dir, "src"), block_sizes=src_blk_sz, edge_probs=src_edge_probs,
                                                  num_channels=feat_dim, centers=centers, flip_y=flip_y, random_state=SEED, train_val_ratio=src_split)


    os.makedirs(os.path.join(data_dir, "tgt"), exist_ok=True)
    tgt_dataset = StochasticBlockModelBlobDataset(root=os.path.join(data_dir, "tgt"), block_sizes=tgt_blk_sz, edge_probs=tgt_edge_probs,
                                                  num_channels=feat_dim, centers=centers, flip_y=flip_y, random_state=SEED, train_val_ratio=tgt_split)



def create_sbm_ls_high_dim(data_dir, src_blk_sz, tgt_blk_sz, src_edge_probs, tgt_edge_probs,
                           n_features, n_informative, n_redundant, n_repeated, flip_y=0.):

    assert src_edge_probs.shape == tgt_edge_probs.shape
    os.makedirs(os.path.join(data_dir, "src"), exist_ok=True)
    kwargs = {"n_informative": n_informative,
              "n_redundant": n_redundant,
              "n_repeated": n_repeated,
              "flip_y": flip_y}
    src_dataset = StochasticBlockModelDataset(root=os.path.join(data_dir, "src"), block_sizes=src_blk_sz, edge_probs=src_edge_probs,
                                              num_channels=n_features, **kwargs)
    os.makedirs(os.path.join(data_dir, "tgt"), exist_ok=True)
    tgt_dataset = StochasticBlockModelDataset(root=os.path.join(data_dir, "tgt"), block_sizes=tgt_blk_sz, edge_probs=tgt_edge_probs,
                                              num_channels=n_features, **kwargs)


if __name__ == "__main__":
    set_model_seed(42)
    root_dir = "/home/hhchung/data/sbm_ls"

    # low-dimension setting
    src_blk_sz = np.array([700, 300])
    tgt_blk_sz = np.array([300, 700])
    edge_probs = np.array([[0.02, 0.002],
                           [0.002, 0.02]])
    centers = np.stack([[np.cos(np.pi / 4),
                         np.sin(np.pi / 4)],
                        [np.cos(5 * np.pi / 4),
                         np.sin(5 * np.pi / 4)]])
    create_sbm_ls_low_dim(os.path.join(root_dir, "low_dim"), src_blk_sz, tgt_blk_sz, edge_probs, edge_probs, centers)


    # high-dimension setting
    # src_blk_sz = np.array([700, 300])
    # tgt_blk_sz = np.array([300, 700])
    # edge_probs = np.array([[0.02, 0.002],
    #                        [0.002, 0.02]])
    # n_features = 128
    # n_informative = 96
    # n_redundant = 16
    # n_repeated = 8
    # create_sbm_ls_high_dim(os.path.join(root_dir, "high_dim"), src_blk_sz, tgt_blk_sz, edge_probs, edge_probs,
    #                        n_features, n_informative, n_redundant, n_repeated)