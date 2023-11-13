import os
import numpy as np
import argparse
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import torch

import sys
sys.path.append("..")
from utils import set_model_seed, get_device
from torch_utils import eval_epoch, create_model
from estimator import BBSELR, BBSETorch, calculate_marginal
from adapter import IWERM, EMBCTS, Tbd

CLASS_NUM = 40


def main(args):
    set_model_seed(args.seed)
    data_src = torch.load(os.path.join(args.data_dir, "ogbn_arxiv", "label_shift", "data_src.pt"))
    data_tgt = torch.load(os.path.join(args.data_dir, "ogbn_arxiv", "label_shift", "data_tgt.pt"))
    test_data_tgt = torch.load(os.path.join(args.data_dir, "ogbn_arxiv", "label_shift", "test_data_tgt.pt"))
    # dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=args.data_dir)
    # data = dataset[0]
    # data.edge_index = to_undirected(data.edge_index, num_nodes=len(data.y))
    # print("Begin partition source data")
    # data_src = temp_partition_arxiv(data, [0, 2012, 2013]) # src train: ~ 2011, src val: 2012
    # print("Begin partition target data")
    # data_tgt = temp_partition_arxiv(data, [2013, 2019, 2020]) # tgt train: 2013 ~ 2018, tgt val: 2019
    # print("Begin partition target test data")
    # test_data_tgt = temp_partition_arxiv(data, [0, 2020, 2021]) # tgt test: 2020
    # test_data_tgt.test_mask = test_data_tgt.val_mask

    device = get_device(args.gpu_id)
    if args.method == "iw-erm" or args.method == "tbd":
        exp_method = args.method + "_" + args.blackbox + "_" + args.estimator
    else:
        exp_method = args.method

    layer_config = str(len(args.model_gnn_dim_list) - 1) + "layer" if args.model not in ["logreg", "mlp"] else ""
    result_subdir = os.path.join(args.result_dir, exp_method, str(args.seed), args.model, layer_config)
    log_subdir = os.path.join(args.log_dir, exp_method, str(args.seed), args.model, layer_config)
    os.makedirs(result_subdir, exist_ok=True)
    os.makedirs(log_subdir, exist_ok=True)

    # importance weight estimation
    if args.estimator == "bbse":
        if args.blackbox == "logreg":
            blackbox = LogisticRegression(random_state=args.seed, max_iter=500)
            bbse = BBSELR(blackbox, data_src, data_tgt, CLASS_NUM)

        else:
            blackbox_hyper_param = {"gnn_dim_list": args.blackbox_gnn_dim_list,
                                    "mlp_dim_list": args.blackbox_mlp_dim_list,
                                    "gnn_dr_list": args.blackbox_gnn_dr_list,
                                    "mlp_dr_list": args.blackbox_mlp_dr_list}
            blackbox = create_model(args.blackbox, device, blackbox_hyper_param)
            bbse = BBSETorch(args.blackbox, args.blackbox_learning_rate, args.blackbox_epochs, blackbox, data_src,
                             data_tgt, CLASS_NUM, device)

        wt, _ = bbse.estimate()
    elif args.method == "erm" or args.method == "em-bcts":
        wt = np.ones(CLASS_NUM)

    # (importance weighted) ERM
    if args.model == "logreg":
        X_src_train, y_src_train_true = data_src.x[data_src.train_mask].numpy(), data_src.y[data_src.train_mask].numpy()
        X_src_val, y_src_val_true = data_src.x[data_src.val_mask].numpy(), data_src.y[data_src.val_mask].numpy()
        X_tgt_test, y_tgt_test_true = test_data_tgt.x[test_data_tgt.val_mask].numpy(), test_data_tgt.y[test_data_tgt.val_mask].numpy()
        class_weights = dict()
        for k in range(len(wt)):
            class_weights[k] = wt[k]
        model = LogisticRegression(random_state=args.seed, class_weight=class_weights)
        model.fit(X_src_train, y_src_train_true)

        y_src_val_pred = model.predict(X_src_val)
        y_tgt_test_pred = model.predict(X_tgt_test)
        y_src_val_prob = model.predict_proba(X_src_val)
        y_tgt_test_prob = model.predict_proba(X_tgt_test)
        src_val_acc = accuracy_score(y_src_val_true, y_src_val_pred)
        src_val_bacc = balanced_accuracy_score(y_src_val_true, y_src_val_pred)
        tgt_test_acc = accuracy_score(y_tgt_test_true, y_tgt_test_pred)
        tgt_test_bacc = balanced_accuracy_score(y_tgt_test_true, y_tgt_test_pred)
    else:
        data_src, data_tgt, test_data_tgt = data_src.to(device), data_tgt.to(device), test_data_tgt.to(device)
        model_hyper_param = {"gnn_dim_list": args.model_gnn_dim_list,
                             "mlp_dim_list": args.model_mlp_dim_list,
                             "gnn_dr_list": args.model_gnn_dr_list,
                             "mlp_dr_list": args.model_mlp_dr_list}
        model = create_model(args.model, device, model_hyper_param)
        if args.method == "tbd":
            y_src_val_true = data_src.y[data_src.val_mask].cpu().numpy()
            y_src_marginal_true = calculate_marginal(y_src_val_true, n_classes=CLASS_NUM)
            y_tgt_train_true = data_tgt.y[data_tgt.train_mask].cpu().numpy() # oracle info
            y_tgt_marginal_true = calculate_marginal(y_tgt_train_true, n_classes=CLASS_NUM) # oracle info
            wt_true = (y_tgt_marginal_true / y_src_marginal_true)
            adapter = Tbd(wt, model, device, y_src_marginal_true, args.tbd_normalizer)
            adapter.adapt(data_src, data_tgt, args.tbd_subgraph_portion_list, args, log_subdir)
        else: # include iwerm, erm, and em-bcts (the latter two use uniform class weights)
            adapter = IWERM(wt, model, device)
            adapter.adapt(data_src, args)
        model = adapter.get_model()

        src_val_loss, src_val_acc, src_val_bacc, y_src_val_pred, y_src_val_prob, y_src_val_true = eval_epoch(model,
                                                                                                             args.model,
                                                                                                             data_src)
        tgt_test_loss, tgt_test_acc, tgt_test_bacc, y_tgt_test_pred, y_tgt_test_prob, y_tgt_test_true = eval_epoch(
            model, args.model, test_data_tgt, test=True)

    # Post-hoc adaptation (EM-BCTS)
    if args.method == "em-bcts":
        adapter = EMBCTS()
        encoder = OneHotEncoder(sparse_output=False)
        y_src_val_true = encoder.fit_transform(y_src_val_true.reshape(-1, 1))
        y_tgt_test_pred = adapter.adapt(y_src_val_true, y_src_val_prob, y_tgt_test_prob)
        # re-evaluate on test set
        tgt_test_acc = accuracy_score(y_tgt_test_true, y_tgt_test_pred)
        tgt_test_bacc = balanced_accuracy_score(y_tgt_test_true, y_tgt_test_pred)

    # print(f"Target - source acc: {tgt_test_acc - src_val_acc}")
    # print(f"Target - source bacc: {tgt_test_bacc - src_val_bacc}")

    result_dict = {"src_acc": src_val_acc,
                   "tgt_acc": tgt_test_acc,
                   "src_bacc": src_val_bacc,
                   "tgt_bacc": tgt_test_bacc,
                   # "src_pred_dist": src_pred_dist,
                   # "tgt_pred_dist": tgt_pred_dist
                   }

    if args.estimator is not None:
        y_src_marginal_true = calculate_marginal(y_src_val_true, n_classes=CLASS_NUM)
        y_tgt_train_true = data_tgt.y[data_tgt.train_mask].cpu().numpy()
        y_tgt_marginal_true = calculate_marginal(y_tgt_train_true, n_classes=CLASS_NUM)
        wt_true = (y_tgt_marginal_true / y_src_marginal_true)
        wt_true = np.squeeze(wt_true)
        # print(f"wt: {wt} wt true: {wt_true}")
        result_dict["wt"] = wt
        result_dict["wt_true"] = wt_true
        result_dict["est_err"] = np.linalg.norm(wt - wt_true)
        result_dict["normalized_est_err"] = np.linalg.norm(wt / wt.sum() - wt_true / wt_true.sum())

    # result_filename = file_name + "_" + (adapter.best_run_name if adapter.best_run_name else "result")
    result_filename = adapter.best_run_name if args.model != "logreg" and adapter.best_run_name else "result"
    pickle.dump(result_dict, open(os.path.join(result_subdir, result_filename + ".pkl"), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="experiments on synthetic datasets")
    parser.add_argument("--method", type=str, help="which method for label shift adaptation", default=None)
    parser.add_argument("--estimator", type=str, help="which label shift estimator to use", default=None)
    parser.add_argument("--blackbox", type=str, help="which model type for blackbox shift estimation", default=None)
    parser.add_argument("--model", type=str, help="which type of model", default="gcn")
    parser.add_argument("--data_dir", type=str, help="root directory of dataset")
    parser.add_argument("--result_dir", type=str, help="directory storing the results", default="result")
    parser.add_argument("--log_dir", type=str, help="directory storing tensorboard log", default="log")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="which gpu to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="which seed to set")

    parser.add_argument("--blackbox_epochs", type=int, default=200,
                        help="number of training epochs for blackbox")
    parser.add_argument("--blackbox_learning_rate", type=float, default=1e-3,
                        help="learning rate of optimizer for blackbox")
    parser.add_argument("--blackbox_gnn_dim_list", type=int, nargs='+',
                        help="feature dimension list of gnn for blackbox")
    parser.add_argument("--blackbox_mlp_dim_list", type=int, nargs='+',
                        help="feature dimension list of mlp for blackbox")
    parser.add_argument("--blackbox_gnn_dr_list", type=float, default=None,
                        nargs='+', help="dropout ratio list of gnn for blackbox")
    parser.add_argument("--blackbox_mlp_dr_list", type=float, default=None,
                        nargs='+', help="dropout ratio list of mlp for blackbox")
    parser.add_argument("--model_epochs", type=int, default=200,
                        help="number of training epochs for final model")
    parser.add_argument("--model_learning_rate", type=float, default=1e-3,
                        help="learning rate of optimizer for final model")
    parser.add_argument("--model_gnn_dim_list", type=int, nargs='+',
                        help="feature dimension list of gnn for final model")
    parser.add_argument("--model_mlp_dim_list", type=int, nargs='+',
                        help="feature dimension list of mlp for final model")
    parser.add_argument("--model_gnn_dr_list", type=float, default=None,
                        nargs='+', help="dropout ratio list of gnn for final model")
    parser.add_argument("--model_mlp_dr_list", type=float, default=None,
                        nargs='+', help="dropout ratio list of mlp for final model")
    parser.add_argument("--tbd_subgraph_portion_list", type=float, default=[0.5],
                        nargs='+', help="list of subsample portions of the graph")
    parser.add_argument("--tbd_normalizer", type=str, default="none",
                        help="normalizer for importance weight product")
    args = parser.parse_args()
    main(args)