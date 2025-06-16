import numpy as np
from scipy.linalg import expm, inv

from .TVOptAlgorithm import TVOptAlgorithm


class L1AO(TVOptAlgorithm):
    def __init__(
        self,
        baseline_alg: TVOptAlgorithm,
        omega: float,
        A_s: np.ndarray,
        T_s: float,
        version: int = 2,
    ):
        self.baseline_alg = baseline_alg
        self.omega = omega
        self.A_s = A_s
        self.T_s = T_s
        self.version = version

        self.const_pre = self._precompute_constant()
        self.p_ad = None

    def state(self, v, grad_est, v_dot_a):
        return {
            "v": np.array(v),
            "grad_est": np.array(grad_est),
            "v_dot_a": np.array(v_dot_a),
        }

    def _update_law_state_predictor(
        self,
        grad_phi_v,
        grad_est,
        grad_phi_vt_est,
        grad_phi_vv,
        v_dot,
    ):
        grad_est_error = grad_est - grad_phi_v
        p_ad = self.p_ad
        if self.version == 1:
            sigma_hat = p_ad
            grad_est_dot = (
                self.A_s @ grad_est_error
                + grad_phi_vt_est
                + grad_phi_vv @ (v_dot + sigma_hat)
            )
        elif self.version == 2:
            h = p_ad["h"]
            grad_est_dot = (
                self.A_s @ grad_est_error + grad_phi_vt_est + grad_phi_vv @ v_dot + h
            )
        return grad_est_dot

    def _precompute_constant(self):
        A_s_inv = inv(self.A_s)
        exp_A_s_T_s = expm(self.A_s * self.T_s)
        tmp = A_s_inv @ (exp_A_s_T_s - np.eye(self.A_s.shape[0]))
        tmp_inv = inv(tmp)
        return tmp_inv @ exp_A_s_T_s

    def _initialize_adaptive_term(self):
        if self.version == 1:
            sigma_hat0 = np.zeros(self.A_s.shape[0])
            return sigma_hat0
        elif self.version == 2:
            h0 = np.zeros(self.A_s.shape[0])
            sigma_hat0 = np.zeros(self.A_s.shape[0])
            return {"h": h0, "sigma_hat": sigma_hat0}

    def update_law_piecewise_constant_adaptation(
        self,
        grad_phi_v,
        grad_est,
        grad_phi_vv,
    ):
        if self.p_ad is None:
            self.p_ad = self._initialize_adaptive_term()
        else:
            tmp_inv_times_exp_A_s_T_s = self.const_pre
            grad_est_error = grad_est - grad_phi_v
            if self.version == 1:
                sigma_hat = -inv(grad_phi_vv) @ (
                    tmp_inv_times_exp_A_s_T_s @ grad_est_error
                )
                self.p_ad = sigma_hat
            elif self.version == 2:
                h = -tmp_inv_times_exp_A_s_T_s @ grad_est_error
                sigma_hat = inv(grad_phi_vv) @ h
                self.p_ad = {"h": h, "sigma_hat": sigma_hat}

    def dynamics(self, X, grad_phi_v, grad_phi_vv, grad_phi_vt_est):
        v = X["v"]
        grad_est = X["grad_est"]
        v_dot_a = X["v_dot_a"]

        # Baseline dynamics:
        v_dot_base = self.baseline_alg.update_law(
            grad_phi_v, grad_phi_vv, grad_phi_vt_est
        )
        v_dot = v_dot_base + v_dot_a

        # Update state predictor.
        grad_est_dot = self._update_law_state_predictor(
            grad_phi_v,
            grad_est,
            grad_phi_vt_est,
            grad_phi_vv,
            v_dot,
        )
        # Update adaptive term.
        if self.version == 1:
            sigma_hat = self.p_ad
        else:
            sigma_hat = self.p_ad["sigma_hat"]

        new_v_dot_a = self.omega * (-sigma_hat - v_dot_a)

        return {"v": v_dot, "grad_est": grad_est_dot, "v_dot_a": new_v_dot_a}

    def update_law(self, grad_phi_v, grad_phi_vv, grad_phi_vt_est):
        return self.baseline_alg.update_law(grad_phi_v, grad_phi_vv, grad_phi_vt_est)
