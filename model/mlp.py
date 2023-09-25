import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dim_list, dropout_list=None):
        super().__init__()
        assert len(dropout_list) == len(dim_list) - 2 # # layer = len(dim_list) - 1 and last layer doesn't need dropout
        self.linears = nn.ModuleList([nn.Linear(dim_list[i], dim_list[i+1]) for i in range(len(dim_list) - 1)])
        self.dim_list = dim_list
        if dropout_list:
            self.dropout_list = dropout_list
        else:
            self.dropout_list = [0. for _ in range(len(self.linears) - 1)]

    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = l(x).relu()
            if i != len(self.linears) - 1:
                x = F.dropout(x, p=self.dropout_list[i], training=self.training)
        return x

# import torch
# mlp = MLP(dim_list=[4,2,1], dropout_list=[0.1])
# x = torch.randn(2, 4)
# print(mlp(x))