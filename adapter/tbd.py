import os
import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

class Tbd:
    def __init__(self, w, model, device, y_src_marginal, normalizer="none"):
        assert normalizer in ["geo-mean", "const-div", "none", "iw-prod", "log"]
        self.w = torch.from_numpy(w).float().to(device)
        self.device = device
        self._set_model(model)

        self.y_src_marginal = torch.from_numpy(y_src_marginal).squeeze().float().to(device)
        self.normalizer = normalizer

        self.best_run_name = None

    def adapt(self, data_src, data_tgt, subgraph_portion_list, args, log_subdir):
        performance_dict = dict()
        for subgraph_portion in subgraph_portion_list:
            run_name = self.normalizer + "_" + str(subgraph_portion)
            self.writer = SummaryWriter(os.path.join(log_subdir, run_name))
            model, score = self._adapt_hp(data_src, data_tgt, subgraph_portion, args)
            performance_dict[subgraph_portion] = {"name": run_name,
                                                  "model": model,
                                                  "score": score}

        best_score = -np.inf
        for subgraph_portion, model_score_dict in performance_dict.items():
            if model_score_dict["score"] > best_score:
                best_score = model_score_dict["score"]
                best_model = model_score_dict["model"]
                best_run_name = model_score_dict["name"]
        self._set_model(best_model)
        self.best_run_name = best_run_name


    def _adapt_hp(self, data_src, data_tgt, subgraph_portion, args):
        model = deepcopy(self.model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.model_learning_rate)
        for i in range(1, args.model_epochs + 1):
            # subsample graph
            subgraph = self._sample_subgraph(data_src, subgraph_portion)
            train_loss = self._train(model, subgraph, optimizer)
            print(f"Iter: {i} Train Loss: {train_loss}")
            self.writer.add_scalar("loss/train", train_loss, i)

            # to remove
            tgt_out = model(data_tgt.x, data_tgt.edge_index)[data_tgt.val_mask]
            tgt_pred = torch.argmax(tgt_out, dim=1).cpu().numpy()
            tgt_acc = accuracy_score(data_tgt.y[data_tgt.val_mask].cpu().numpy(), tgt_pred)
            self.writer.add_scalar("target acc/val", tgt_acc, i)


        # how to validate and do hp search? use target data w/ validator?
        tgt_out = model(data_tgt.x, data_tgt.edge_index)[data_tgt.val_mask]
        # validator score
        # cheat for a bit
        tgt_pred = torch.argmax(tgt_out, dim=1).cpu().numpy()
        tgt_acc = accuracy_score(data_tgt.y[data_tgt.val_mask].cpu().numpy(), tgt_pred)
        score = tgt_acc
        print(f"Score: {score}")
        return model, score

    def _train(self, model, data, optimizer):
        model.train()
        out = model(data.x, data.edge_index)
        if "train_mask" in data:
            mask = data.train_mask
        else:
            mask = torch.ones_like(data.y, dtype=torch.bool) # subsampled graph contains only training nodes and we don't include train mask
        loss = F.nll_loss(F.log_softmax(out[mask], dim=1), data.y[mask])
        subgraph_nodenum = mask.sum().item()
        if self.normalizer == "geo-mean":
            loss *= torch.prod(self.w[data.y[mask]] ** (1 / subgraph_nodenum)) # importance weighted loss
        elif self.normalizer == "const-div":
            per_node_normalizer = torch.prod(self.w ** (self.y_src_marginal))
            loss *= torch.prod(self.w[data.y[mask]] / per_node_normalizer)
        elif self.normalizer == "none":
            loss *= torch.prod(self.w[data.y[mask]])
        elif self.normalizer == "iw-prod":
            pass
        elif self.normalizer == "log":
            loss = torch.log(loss) + torch.sum(torch.log(self.w[data.y[mask]]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def _sample_subgraph(self, data, p):
        total_node_num = data.x.shape[0]
        train_node_num = data.train_mask.sum().item()
        subgraph_node_num = int(train_node_num * p)
        train_node_index = torch.argwhere(data.train_mask).squeeze()
        subset, _ = torch.sort(train_node_index[torch.randperm(train_node_num)[:subgraph_node_num]])
        edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=total_node_num)
        return Data(x=data.x[subset], edge_index=edge_index, y=data.y[subset]).to(self.device)


    def _set_model(self, model):
        self.model = deepcopy(model)

    def get_model(self):
        return self.model

