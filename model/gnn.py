import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN(nn.Module):
    def __init__(self, dim_list, dropout_list=None):
        super().__init__()
        if dropout_list:
            assert len(dropout_list) == len(
                dim_list) - 2  # # layer = len(dim_list) - 1 and last layer doesn't need dropout
            self.dropout_list = dropout_list
        else:
            self.dropout_list = [0. for _ in range(len(dim_list) - 2)]
        self.gcnconvs = nn.ModuleList([GCNConv(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.gcnconvs):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i != len(self.gcnconvs) - 1:
                x = F.dropout(x, p=self.dropout_list[i], training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.gcnconvs:
            conv.reset_parameters()


class GraphSAGE(nn.Module):
    def __init__(self, dim_list, dropout_list=None):
        super().__init__()
        if dropout_list:
            assert len(dropout_list) == len(
                dim_list) - 2  # # layer = len(dim_list) - 1 and last layer doesn't need dropout
            self.dropout_list = dropout_list
        else:
            self.dropout_list = [0. for _ in range(len(dim_list) - 2)]
        self.sageconvs = nn.ModuleList([SAGEConv(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])


    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.sageconvs):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i != len(self.sageconvs) - 1:
                x = F.dropout(x, p=self.dropout_list[i], training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.sageconvs:
            conv.reset_parameters()


class GAT(nn.Module):
    def __init__(self, dim_list, heads_list=None, dropout_list=None, gat_dropout_list=None):
        super().__init__()
        if heads_list:
            assert len(heads_list) == len(dim_list) - 1  # same as # layer
            self.heads_list = [1] + heads_list
        else:
            self.heads_list = [1] + [1 for _ in range(len(dim_list) - 1)]
        if dropout_list:
            assert len(dropout_list) == len(dim_list) - 2
            self.dropout_list = dropout_list
        else:
            self.dropout_list = [0. for _ in range(len(dim_list) - 2)]
        if gat_dropout_list:
            assert len(gat_dropout_list) == len(dim_list) - 1  # same as # layer
            self.gat_dropout_list = gat_dropout_list
        else:
            self.gat_dropout_list = [0. for _ in range(len(dim_list) - 1)]

        self.gatconvs = nn.ModuleList([GATConv(dim_list[i] * self.heads_list[i], dim_list[i+1], self.heads_list[i+1], dropout=self.gat_dropout_list[i]) for i in range(len(dim_list) - 1)])

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.gatconvs):
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            if i != len(self.gatconvs) - 1:
                x = F.dropout(x, p=self.dropout_list[i], training=self.training)
        return x

    def reset_parameters(self):
        for conv in self.gatconvs:
            conv.reset_parameters()


class LinearGCN(nn.Module):
    def __init__(self, dim_list):
        assert len(dim_list) >= 2
        super().__init__()
        self.gcnconvs = nn.ModuleList([GCNConv(dim_list[i], dim_list[i + 1]) for i in range(len(dim_list) - 1)])

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.gcnconvs):
            x = conv(x, edge_index, edge_weight)
        return x

    def reset_parameters(self):
        for conv in self.gcnconvs:
            conv.reset_parameters()

# import torch
# from torch_geometric.data import Data
#
# edge_index = torch.tensor([[0, 1],
#                            [1, 0],
#                            [1, 2],
#                            [2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
# data = Data(x=x, edge_index=edge_index.t().contiguous())
#
#
# dim_list = [1, 2, 1]
# heads_list = [2, 2]
# dropout_list = [0.1]
# gat_dropout_list = [0.2, 0.2]
# gnn = GAT(dim_list, heads_list=heads_list, dropout_list=dropout_list, gat_dropout_list=gat_dropout_list)
# print(gnn(data.x, data.edge_index))