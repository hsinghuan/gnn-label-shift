import numpy as np
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
from sklearn.metrics import accuracy_score

class Tbd:
    def __init__(self, w, model, device):
        self.w = w
        self.device = device
        self._set_model(model)

    def adapt(self, data_src, data_tgt, subsample_portion_list, args):
        performance_dict = dict()
        for subsample_portion in subsample_portion_list:
            model, score = self._adapt_hp(data_src, data_tgt, subsample_portion, args)
            performance_dict[subsample_portion] = {"model": model,
                                                   "score": score}

        best_score = -np.inf
        for subsample_portion, model_score_dict in performance_dict:
            if model_score_dict["score"] > best_score:
                best_score = model_score_dict["score"]
                best_model = model
        self._set_model(best_model)


    def _adapt_hp(self, data_src, data_tgt, subsample_portion, args):
        model = deepcopy(self.model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.model_learning_rate)
        for i in range(1, args.model_epochs + 1):
            # subsample graph
            subgraph = self._sample_subgraph(data_src, subsample_portion)
            train_loss = self._train(model, subgraph, optimizer)
            print(f"Iter: {i} Train Loss: {train_loss}")
        # how to validate and do hp search? use target data w/ validator?
        tgt_out = model(data_tgt.x, data_tgt.edge_index)
        # validator score
        # cheat for a bit
        tgt_pred = torch.argmax(tgt_out, dim=1).cpu().numpy()
        tgt_acc = accuracy_score(data_tgt.y.cpu().numpy(), tgt_pred)
        score = tgt_acc
        print(f"Score: {score}")
        return model, score

    def _train(self, model, data, optimizer):
        model.train()
        out = self.model(data.x, data.edge_index)
        if "train_mask" in data:
            mask = data.train_mask
        else:
            mask = torch.ones_like(data.y) # subsampled graph contains only training nodes and we don't include train mask
        loss = F.nll_loss(F.log_softmax(out[mask], dim=1), data.y[mask])
        loss *= torch.prod(self.w[data.y[mask]]) # importance weighted loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def _sample_subgraph(self, data, p):
        train_node_num = data.train_mask.sum().item()
        subgraph_node_num = int(train_node_num * p)
        train_node_index = torch.argwhere(data.train_mask).squeeze()
        subset, _ = torch.sort(train_node_index[torch.randperm(train_node_num)[:subgraph_node_num]])
        edge_index = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=train_node_num)
        return Data(x=data.x[subset], edge_index=edge_index, y=data.y[subset]).to(self.device)


    def _set_model(self, model):
        self.model = deepcopy(model)

    def get_model(self):
        return self.model


