import numpy as np
from sklearn.metrics import confusion_matrix

import utils

class BBSE():
    def __init__(self):
        pass

    def estimate_importance_weight(self, y_true_src, y_pred_src, y_pred_tgt, n_classes):

        labels = np.arange(n_classes)
        C = confusion_matrix(y_true_src, y_pred_src, labels=labels).T
        C = C / y_true_src.shape[0]

        mu_t = utils.calculate_marginal(y_pred_tgt, n_classes)
        lamb = 1.0 / min(y_pred_src.shape[0], y_pred_tgt.shape[0])

        I = np.eye(n_classes)
        wt = np.linalg.solve(np.dot(C.T, C) + lamb * I, np.dot(C.T, mu_t))
        return wt

    def estimate_target_dist(self, wt, y_true_src, n_classes):
        mu_t = utils.calculate_marginal(y_true_src, n_classes)
        return wt * mu_t


# y_true_src = np.array([2, 0, 2, 2, 0, 1])
# y_pred_src = np.array([0, 0, 2, 2, 0, 2])
# y_pred_tgt = np.array([0,1,1,1,2,2,1,1,1,1,1,2,2,0])
# bbse = BBSE()
# print(utils.calculate_marginal(y_true_src, n_classes=3))
# print(utils.calculate_marginal(y_pred_tgt, n_classes=3))
# wt = bbse.estimate_importance_weight(y_true_src, y_pred_src, y_pred_tgt, 3)
# print(wt)
# tgt_dist = bbse.estimate_target_dist(wt, y_true_src, 3)
# print(tgt_dist)