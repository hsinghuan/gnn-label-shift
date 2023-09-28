"""
Experiment script for changing inter-class connection probability under label shift
to answer the question: how does homophily affect gcn's reaction to label shift
"""
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch_geometric.utils import homophily

from dataset import load_sbm_dataset
import sys
sys.path.append("..")
from model import GCN, MLP
from utils import set_model_seed, get_device

def train_epoch(encoder, mlp, data, optimizer):
    encoder.train()
    mlp.train()
    out = mlp(encoder(data.x, data.edge_index))
    loss = F.nll_loss(F.log_softmax(out[data.train_mask], dim=1), data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def eval_epoch(encoder, mlp, data, test=False):
    encoder.eval()
    mlp.eval()
    out = mlp(encoder(data.x, data.edge_index))
    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask
    loss = F.nll_loss(F.log_softmax(out[mask], dim=1), data.y[mask])
    y_pred = torch.argmax(out[mask], dim=1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    return loss.item(), acc

@torch.no_grad()
def observe_pred_dist(encoder, mlp, data_src, data_tgt):
    encoder.eval()
    mlp.eval()
    src_out = mlp(encoder(data_src.x, data_src.edge_index))
    y_src_val_pred = torch.argmax(src_out[data_src.val_mask], dim=1).cpu().numpy()
    y_src_val_true = data_src.y[data_src.val_mask].cpu().numpy()

    tgt_out = mlp(encoder(data_tgt.x, data_tgt.edge_index))
    y_tgt_test_pred = torch.argmax(tgt_out[data_tgt.test_mask], dim=1).cpu().numpy()
    y_tgt_test_true = data_tgt.y[data_tgt.test_mask].cpu().numpy()

    y_dist_src_val_true, _ = np.histogram(y_src_val_true, bins=2)
    y_dist_src_val_true = np.divide(y_dist_src_val_true, y_dist_src_val_true.sum())

    y_dist_src_val_pred, _ = np.histogram(y_src_val_pred, bins=2)
    y_dist_src_val_pred = np.divide(y_dist_src_val_pred, y_dist_src_val_pred.sum())
    src_total_variation = (y_dist_src_val_true - y_dist_src_val_pred).max()

    y_dist_tgt_test_true, _ = np.histogram(y_tgt_test_true, bins=2)
    y_dist_tgt_test_true = np.divide(y_dist_tgt_test_true, y_dist_tgt_test_true.sum())

    y_dist_tgt_test_pred, _ = np.histogram(y_tgt_test_pred, bins=2)
    y_dist_tgt_test_pred = np.divide(y_dist_tgt_test_pred, y_dist_tgt_test_pred.sum())

    tgt_total_variation = (y_dist_tgt_test_true - y_dist_tgt_test_pred).max()

    return src_total_variation, tgt_total_variation

def main(args):
    set_model_seed(args.seed)
    device = get_device(args.gpu_id)
    encoder = GCN(args.gnn_dim_list, dropout_list=args.gnn_dr_list).to(device)
    mlp = MLP(args.mlp_dim_list, dropout_list=args.mlp_dr_list).to(device)
    if args.exp_name == "vary_block_prob":
        data_src = load_sbm_dataset(os.path.join(args.data_dir, args.exp_name, args.exp_param), src=True).to(device)
    elif args.exp_name == "vary_shift":
        data_src = load_sbm_dataset(os.path.join(args.data_dir, args.exp_name), src=True).to(device)
    data_tgt = load_sbm_dataset(os.path.join(args.data_dir, args.exp_name, args.exp_param), src=False).to(device)
    optimizer = torch.optim.Adam(params=list(encoder.parameters()) + list(mlp.parameters()), lr=args.learning_rate)

    for e in range(1, args.epochs + 1):
        train_loss = train_epoch(encoder, mlp, data_src, optimizer)
        val_loss, val_acc = eval_epoch(encoder, mlp, data_src)
        print(f"Epoch: {e} Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_acc}")

    src_homophily = homophily(data_src.edge_index, data_src.y, method="edge")
    tgt_homophily = homophily(data_tgt.edge_index, data_tgt.y, method="edge")

    src_val_loss, src_val_acc = eval_epoch(encoder, mlp, data_src)
    print(f"Source Val Acc: {src_val_acc}")

    tgt_test_loss, tgt_test_acc = eval_epoch(encoder, mlp, data_tgt, test=True)
    print(f"Target Test Acc: {tgt_test_acc}")

    src_tv, tgt_tv = observe_pred_dist(encoder, mlp, data_src, data_tgt)
    print(f"Source homophily: {src_homophily} acc: {src_val_acc} tv: {src_tv}")
    print(f"Target homophily: {tgt_homophily} acc: {tgt_test_acc} tv: {tgt_tv}")
    print(f"Target - source acc: {tgt_test_acc - src_val_acc}")

    result_dict = {"src_acc": src_val_acc,
                   "tgt_acc": tgt_test_acc,
                   "src_homophily": src_homophily,
                   "tgt_homophily": tgt_homophily,
                   "src_tv": src_tv,
                   "tgt_tv": tgt_tv
                   }



    result_subdir = os.path.join(args.result_dir, args.exp_name, str(args.seed))
    os.makedirs(result_subdir, exist_ok=True)
    result_filename = args.exp_param + ".pkl"
    pickle.dump(result_dict, open(os.path.join(result_subdir, result_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiments on synthetic datasets")
    parser.add_argument("--data_dir", type=str, help="root directory of dataset")
    parser.add_argument("--exp_name", type=str, help="name of the experiment, e.g. vary_block_prob", default="vary_block_prob")
    parser.add_argument("--exp_param", type=str, help="experiment parameter, e.g. 0.003", default="")
    parser.add_argument("--result_dir", type=str, help="directory storing the results", default="result")
    parser.add_argument("--gnn_dim_list", type=int, nargs='+',
                        help="feature dimension list of gnn")
    parser.add_argument("--mlp_dim_list", type=int, nargs='+',
                        help="feature dimension list of mlp")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="learning rate of optimizer")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="which gpu to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="which seed to set")
    parser.add_argument("--gnn_dr_list", type=float, default=None,
                        nargs='+', help="dropout ratio list of gnn")
    parser.add_argument("--mlp_dr_list", type=float, default=None,
                        nargs='+', help="dropout ratio list of mlp")
    args = parser.parse_args()
    main(args)