import numpy as np
from copy import deepcopy
import torch
import sys
sys.path.append("..")
from torch_utils import torch_fit

class IWERM:
    def __init__(self, w, model, device):
        self.w = w
        self.device = device
        self._set_model(model)

    def adapt(self, data_src, args):
        wt_tensor = torch.from_numpy(self.w).float().to(self.device)
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.model_learning_rate)
        torch_fit(self.model, args.model, data_src, optimizer, args.model_epochs, class_weight=wt_tensor)

    def _set_model(self, model):
        self.model = deepcopy(model)

    def get_model(self):
        return self.model

