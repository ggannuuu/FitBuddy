import numpy as np
from scipy.linalg import inv

from .TVOptAlgorithm import TVOptAlgorithm


class PCIP(TVOptAlgorithm):
    def __init__(self, P, eps=None):
        self.P = P
        self.eps = eps

    def update_law(self, grad_phi_v, grad_phi_vv, grad_phi_vt_est):
        if self.eps is None:
            v_dot = -inv(grad_phi_vv) @ (grad_phi_vt_est + self.P @ grad_phi_v)  # PCIP
        else:  # Modified PCIP
            norm_grad = np.linalg.norm(grad_phi_v)
            v_dot = -inv(grad_phi_vv) @ (
                grad_phi_vt_est + self.P @ grad_phi_v / max(norm_grad, self.eps)
            )
        return v_dot

    def dynamics(self, X, grad_phi_v, grad_phi_vv, grad_phi_vt_est):
        return {"v": self.update_law(grad_phi_v, grad_phi_vv, grad_phi_vt_est)}

    def state(self, v):
        return {"v": np.array(v)}
