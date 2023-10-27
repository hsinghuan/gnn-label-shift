import numpy as np

from abstention.calibration import TempScaling
from abstention.label_shift import EMImbalanceAdapter

class EMBCTS:
    def __init__(self):
        bcts_calibrator_factory = TempScaling(verbose=False, bias_positions="all")
        self.adapter = EMImbalanceAdapter(calibrator_factory=bcts_calibrator_factory)

    def adapt(self, y_src_true, y_src_prob, y_tgt_prob):
        # print("y src true", y_src_true)
        # print("y tgt prob", y_tgt_prob)
        # print("y src prob", y_src_prob)
        adapter_fn = self.adapter(valid_labels=y_src_true,
                                  tofit_initial_posterior_probs=y_tgt_prob,
                                  valid_posterior_probs=y_src_prob)
        adapted_y_tgt_prob = adapter_fn(y_tgt_prob)

        return np.argmax(adapted_y_tgt_prob, axis=1)