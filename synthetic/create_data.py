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





if __name__ == "__main__":
    set_model_seed(42)
    root_dir = "/home/hhchung/data/sbm_ls"

    # vary block probability (homophily)
    src_blk_sz = np.array([700, 300])
    tgt_blk_sz = np.array([300, 700])
    centers = np.stack([[np.cos(np.pi / 4),
                         np.sin(np.pi / 4)],
                        [np.cos(5 * np.pi / 4),
                         np.sin(5 * np.pi / 4)]])
    intra_edge_prob = 0.02
    inter_edge_prob_list = [0.003, 0.006, 0.009, 0.012, 0.015]
    for p in inter_edge_prob_list:
        edge_probs = np.array([[intra_edge_prob, p],
                               [p, intra_edge_prob]])
        create_sbm_ls_low_dim(os.path.join(root_dir, "vary_block_prob", str(p)), src_blk_sz, tgt_blk_sz, edge_probs,
                              edge_probs, centers)

    # vary shift intensity
    src_blk_sz = np.array([700, 300])
    centers = np.stack([[np.cos(np.pi / 4),
                         np.sin(np.pi / 4)],
                        [np.cos(5 * np.pi / 4),
                         np.sin(5 * np.pi / 4)]])
    edge_probs = np.array([[0.02, 0.009],
                           [0.009, 0.02]])
    tgt_class_0_shift_list = [-0.2, -0.3, -0.4, -0.5, -0.6]
    for shift in tgt_class_0_shift_list:
        tgt_blk_sz[0] = src_blk_sz[0] + shift * src_blk_sz.sum()
        tgt_blk_sz[1] = src_blk_sz.sum() - tgt_blk_sz[0]
        create_sbm_ls_low_dim(os.path.join(root_dir, "vary_shift", str(shift)), src_blk_sz, tgt_blk_sz, edge_probs,
                              edge_probs, centers)
