"""
Experiment script for changing inter-class connection probability under label shift
to answer the question: how does homophily affect gcn's reaction to label shift
"""
import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn.functional as F
from torch_geometric.utils import homophily

from dataset import load_sbm_dataset
import sys
sys.path.append("..")
from model import GCN, LinearGCN, MLP, Model
from utils import set_model_seed, get_device

def train_epoch(model, data, optimizer):
    model.train()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(F.log_softmax(out[data.train_mask], dim=1), data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def eval_epoch(model, data, test=False):
    model.eval()
    out = model(data.x, data.edge_index)
    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask
    loss = F.nll_loss(F.log_softmax(out[mask], dim=1), data.y[mask])
    y_pred = torch.argmax(out[mask], dim=1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return loss.item(), acc, bacc, y_pred, y_true

@torch.no_grad()
def calc_tv(y_true, y_pred, num_class=2):

    y_true_dist, _ = np.histogram(y_true, bins=num_class)
    y_true_dist = np.divide(y_true_dist, y_true_dist.sum())

    y_pred_dist, _ = np.histogram(y_pred, bins=num_class)
    y_pred_dist = np.divide(y_pred_dist, y_pred_dist.sum())

    tv = np.abs(y_true_dist - y_pred_dist).max()
    return tv, y_pred_dist


def main(args):
    set_model_seed(args.seed)
    if args.exp_name == "vary_block_prob":
        data_src = load_sbm_dataset(os.path.join(args.data_dir, args.exp_name, args.exp_param), src=True)
    elif args.exp_name == "vary_shift":
        data_src = load_sbm_dataset(os.path.join(args.data_dir, args.exp_name), src=True)
    data_tgt = load_sbm_dataset(os.path.join(args.data_dir, args.exp_name, args.exp_param), src=False)

    src_homophily = homophily(data_src.edge_index, data_src.y, method="edge")
    tgt_homophily = homophily(data_tgt.edge_index, data_tgt.y, method="edge")

    if args.model == "logreg":
        X_src_train, y_src_train_true = data_src.x[data_src.train_mask], data_src.y[data_src.train_mask]
        X_src_val, y_src_val_true = data_src.x[data_src.val_mask], data_src.y[data_src.val_mask]
        X_tgt_test, y_tgt_test_true = data_tgt.x[data_tgt.test_mask], data_tgt.y[data_tgt.test_mask]
        model = LogisticRegression(random_state=args.seed)
        model.fit(X_src_train, y_src_train_true)
        y_src_val_pred = model.predict(X_src_val)
        y_tgt_test_pred = model.predict(X_tgt_test)
        src_val_acc = accuracy_score(y_src_val_true, y_src_val_pred)
        src_val_bacc = balanced_accuracy_score(y_src_val_true, y_src_val_pred)
        tgt_test_acc = accuracy_score(y_tgt_test_true, y_tgt_test_pred)
        tgt_test_bacc = balanced_accuracy_score(y_tgt_test_true, y_tgt_test_pred)


    else:
        device = get_device(args.gpu_id)
        data_src, data_tgt = data_src.to(device), data_tgt.to(device)
        if args.model == "gcn":
            encoder = GCN(args.gnn_dim_list, dropout_list=args.gnn_dr_list)
            mlp = MLP(args.mlp_dim_list, dropout_list=args.mlp_dr_list)
            model = Model(encoder, mlp).to(device)
        elif args.model == "lingcn":
            model = LinearGCN(args.gnn_dim_list).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

        for e in range(1, args.epochs + 1):
            train_loss = train_epoch(model, data_src, optimizer)
            val_loss, val_acc, val_bacc, _, _ = eval_epoch(model, data_src)
            print(f"Epoch: {e} Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_acc} Val Bacc: {val_bacc}")



        src_val_loss, src_val_acc, src_val_bacc, y_src_val_pred, y_src_val_true = eval_epoch(model, data_src)

        tgt_test_loss, tgt_test_acc, tgt_test_bacc, y_tgt_test_pred, y_tgt_test_true = eval_epoch(model, data_tgt, test=True)

        # with torch.no_grad():
        #     model.eval()
        #     src_out = model(data_src.x, data_src.edge_index)
        #     y_src_val_pred = torch.argmax(src_out[data_src.val_mask], dim=1).cpu().numpy()
        #     y_src_val_true = data_src.y[data_src.val_mask].cpu().numpy()
        #
        #     tgt_out = model(data_tgt.x, data_tgt.edge_index)
        #     y_tgt_test_pred = torch.argmax(tgt_out[data_tgt.test_mask], dim=1).cpu().numpy()
        #     y_tgt_test_true = data_tgt.y[data_tgt.test_mask].cpu().numpy()

    src_tv, src_pred_dist = calc_tv(y_src_val_true, y_src_val_pred, num_class=2)
    tgt_tv, tgt_pred_dist = calc_tv(y_tgt_test_true, y_tgt_test_pred, num_class=2)

    print(f"Source homophily: {src_homophily} acc: {src_val_acc} bacc:{src_val_bacc} tv: {src_tv} pred dist: {src_pred_dist}")
    print(f"Target homophily: {tgt_homophily} acc: {tgt_test_acc} bacc:{tgt_test_bacc} tv: {tgt_tv} pred dist: {tgt_pred_dist}")
    print(f"Target - source acc: {tgt_test_acc - src_val_acc}")

    result_dict = {"src_acc": src_val_acc,
                   "tgt_acc": tgt_test_acc,
                   "src_bacc": src_val_bacc,
                   "tgt_bacc": tgt_test_bacc,
                   "src_homophily": src_homophily,
                   "tgt_homophily": tgt_homophily,
                   "src_tv": src_tv,
                   "tgt_tv": tgt_tv,
                   "src_pred_dist": src_pred_dist,
                   "tgt_pred_dist": tgt_pred_dist
                   }


    layer_config = str(len(args.gnn_dim_list) - 1) + "layer" if args.model != "logreg" else ""
    result_subdir = os.path.join(args.result_dir, args.exp_name, args.model, layer_config, str(args.seed))
    os.makedirs(result_subdir, exist_ok=True)
    result_filename = args.exp_param + ".pkl"
    pickle.dump(result_dict, open(os.path.join(result_subdir, result_filename), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiments on synthetic datasets")
    parser.add_argument("--data_dir", type=str, help="root directory of dataset")
    parser.add_argument("--exp_name", type=str, help="name of the experiment, e.g. vary_block_prob", default="vary_block_prob")
    parser.add_argument("--exp_param", type=str, help="experiment parameter, e.g. 0.003", default="")
    parser.add_argument("--result_dir", type=str, help="directory storing the results", default="result")
    parser.add_argument("--model", type=str, help="which type of model", default="gcn")
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