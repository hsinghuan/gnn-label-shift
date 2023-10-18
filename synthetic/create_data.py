import numpy as np
import os

from dataset import StochasticBlockModelBlobDataset
import sys
sys.path.append("..")
from utils import set_model_seed

SEED = 42

def create_sbm_ls(data_dir, centers, cluster_std, src_blk_sz=None, tgt_blk_sz=None, src_edge_probs=None, tgt_edge_probs=None, flip_y=0.,
                          src_split=[0.8,0.2], tgt_split=[0.6,0.2]):
    """
    """
    class_num = centers.shape[0]
    feat_dim = centers.shape[1]
    if src_edge_probs is not None:
        assert src_edge_probs.shape[0] == class_num
    if tgt_edge_probs is not None:
        assert tgt_edge_probs.shape[0] == class_num
    if src_blk_sz is not None and src_edge_probs is not None:
        os.makedirs(os.path.join(data_dir, "src"), exist_ok=True)
        src_dataset = StochasticBlockModelBlobDataset(root=os.path.join(data_dir, "src"), block_sizes=src_blk_sz, edge_probs=src_edge_probs,
                                                      num_channels=feat_dim, centers=centers, cluster_std=cluster_std, flip_y=flip_y, random_state=SEED, train_val_ratio=src_split)

    if tgt_blk_sz is not None and tgt_edge_probs is not None:
        os.makedirs(os.path.join(data_dir, "tgt"), exist_ok=True)
        tgt_dataset = StochasticBlockModelBlobDataset(root=os.path.join(data_dir, "tgt"), block_sizes=tgt_blk_sz, edge_probs=tgt_edge_probs,
                                                      num_channels=feat_dim, centers=centers, cluster_std=cluster_std, flip_y=flip_y, random_state=SEED, train_val_ratio=tgt_split)





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
    cluster_std = 1 / np.sqrt(2)
    intra_edge_prob = 0.02
    inter_edge_prob_list = [0.003, 0.0045, 0.006, 0.0075, 0.009,  0.0105, 0.012, 0.0135, 0.015, 0.0165, 0.0180, 0.0195]
    for p in inter_edge_prob_list:
        edge_probs = np.array([[intra_edge_prob, p],
                               [p, intra_edge_prob]])
        create_sbm_ls(os.path.join(root_dir, "vary_block_prob", str(p)), centers, cluster_std, src_blk_sz, tgt_blk_sz, edge_probs,
                              edge_probs)

    # vary shift intensity
    src_blk_sz = np.array([700, 300])
    centers = np.stack([[np.cos(np.pi / 4),
                         np.sin(np.pi / 4)],
                        [np.cos(5 * np.pi / 4),
                         np.sin(5 * np.pi / 4)]])
    cluster_std = 1 / np.sqrt(2)
    edge_probs = np.array([[0.02, 0.009],
                           [0.009, 0.02]])
    # create source dataset
    create_sbm_ls(os.path.join(root_dir, "vary_shift"), centers, cluster_std, src_blk_sz=src_blk_sz, src_edge_probs=edge_probs)

    # create target dataset
    tgt_class_0_shift_list = [-0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
    for shift in tgt_class_0_shift_list:
        tgt_blk_sz[0] = src_blk_sz[0] + shift * src_blk_sz.sum()
        tgt_blk_sz[1] = src_blk_sz.sum() - tgt_blk_sz[0]
        create_sbm_ls(os.path.join(root_dir, "vary_shift", str(shift)), centers, cluster_std, tgt_blk_sz=tgt_blk_sz, tgt_edge_probs=edge_probs)
