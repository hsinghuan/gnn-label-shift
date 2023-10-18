import numpy as np
import torch.optim
from sklearn.metrics import confusion_matrix

from .utils import calculate_marginal
import sys
sys.path.append("..")
from torch_utils import torch_fit, forward

class BBSE:
    def __init__(self, blackbox, data_src, data_tgt, n_classes, device=None):
        self.blackbox = blackbox
        self.data_src = data_src
        self.data_tgt = data_tgt
        if device:
            self.data_src = self.data_src.to(device)
            self.data_tgt = self.data_tgt.to(device)
        self.n_classes = n_classes
        self.device = device

    def estimate(self):
        y_true_src, y_pred_src, y_pred_tgt = self._train_pred_blackbox()
        wt = self._estimate_importance_weight(y_true_src, y_pred_src, y_pred_tgt)
        y_marginal_tgt = self._estimate_target_dist(wt, y_true_src)
        wt = np.squeeze(wt)
        y_marginal_tgt = np.squeeze(y_marginal_tgt)
        return wt, y_marginal_tgt

    def _train_pred_blackbox(self):
        raise NotImplementedError("Subclass should implement this")

    def _estimate_importance_weight(self, y_true_src, y_pred_src, y_pred_tgt):
        labels = np.arange(self.n_classes)
        C = confusion_matrix(y_true_src, y_pred_src, labels=labels).T
        C = C / y_true_src.shape[0]

        mu_t = calculate_marginal(y_pred_tgt, self.n_classes)
        lamb = 1.0 / min(y_pred_src.shape[0], y_pred_tgt.shape[0])

        I = np.eye(self.n_classes)
        wt = np.linalg.solve(np.dot(C.T, C) + lamb * I, np.dot(C.T, mu_t))
        return wt

    def _estimate_target_dist(self, wt, y_true_src):
        mu_t = calculate_marginal(y_true_src, self.n_classes)
        return wt * mu_t


class BBSELR(BBSE):
    def __init__(self, blackbox, data_src, data_tgt, n_classes):
        super().__init__(blackbox, data_src, data_tgt, n_classes)

    def _train_pred_blackbox(self):
        X_src_train, y_src_train_true = self.data_src.x[self.data_src.train_mask].numpy(), self.data_src.y[self.data_src.train_mask].numpy()
        X_src_val, y_src_val_true = self.data_src.x[self.data_src.val_mask].numpy(), self.data_src.y[self.data_src.val_mask].numpy()
        self.blackbox.fit(X_src_train, y_src_train_true)
        y_src_val_pred = self.blackbox.predict(X_src_val)
        X_tgt_train = self.data_tgt.x[self.data_tgt.train_mask]
        y_tgt_train_pred = self.blackbox.predict(X_tgt_train)

        return y_src_val_true, y_src_val_pred, y_tgt_train_pred


class BBSETorch(BBSE):
    def __init__(self, model_name, lr, epochs, blackbox, data_src, data_tgt, n_classes, device):
        super().__init__(blackbox, data_src, data_tgt, n_classes, device)
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs

    def _train_pred_blackbox(self):
        optimizer = torch.optim.Adam(self.blackbox.parameters(), lr=self.lr)
        torch_fit(self.blackbox, self.model_name, self.data_src, optimizer, self.epochs)
        out_src = forward(self.blackbox, self.model_name, self.data_src, eval=True)
        y_src_val_pred = torch.argmax(out_src[self.data_src.val_mask], dim=1).cpu().numpy()
        out_tgt = forward(self.blackbox, self.model_name, self.data_tgt, eval=True)
        y_tgt_train_pred = torch.argmax(out_tgt[self.data_tgt.train_mask], dim=1).cpu().numpy()
        y_src_val_true = self.data_src.y[self.data_src.val_mask].cpu().numpy()

        return y_src_val_true, y_src_val_pred, y_tgt_train_pred