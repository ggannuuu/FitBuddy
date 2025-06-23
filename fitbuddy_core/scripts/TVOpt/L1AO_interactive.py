import math

import torch


class L1AO_interactive:
    def __init__(
        self,
        baseline_alg,
        omega: float,
        A_s: torch.Tensor,
        T_s: float = None,  # if specified, the adaptation law will be applied with the fixed time step
    ):
        self.baseline_alg = baseline_alg
        self.omega = omega
        self.A_s = A_s

        self.grad_est = None
        self.v_dot_a = None
        self.h = None
        self.sigma_hat = None
        self.p0 = None
        self.p = None
        self.p_dot0 = None
        self.p_dot = None
        self.ad_first = True
        self.T_s = T_s

    def _precompute_constant(self, T_s):
        exp_A_s_T_s = torch.matrix_exp(self.A_s * T_s)
        # tmp = torch.linalg.inv((torch.eye(self.A_s.shape[0]) - exp_A_s_T_s)) @ self.A_s
        tmp = torch.empty_like(self.A_s)
        torch.linalg.solve(
            torch.eye(self.A_s.shape[0], dtype=self.A_s.dtype) - exp_A_s_T_s,
            self.A_s @ exp_A_s_T_s,
            out=tmp,
        )
        return tmp

    def initialize(self, v0, t0):
        self.baseline_alg.initialize(v0, t0)
        self._initialize()

    def _initialize(self):
        # for adaptation
        self.v_dot_a = torch.zeros(self.A_s.shape[0], dtype=self.A_s.dtype)
        self.h = torch.zeros(self.A_s.shape[0], dtype=self.A_s.dtype)
        self.sigma_hat = torch.zeros(self.A_s.shape[0], dtype=self.A_s.dtype)
        self.ad_first = True
        self.grad_est = None
        if self.T_s is not None:
            self.const_pre = self._precompute_constant(self.T_s)

    def reset(self):
        self.baseline_alg.reset()
        self._initialize()

    def _update_law_state_predictor(
        self,
        grad_est,
        grad_v_Phi,
        grad_vv_Phi,
        grad_vt_Phi_est,
        v_dot,
    ):
        grad_est_error = grad_est - grad_v_Phi
        h = self.h
        grad_est_dot = (
            self.A_s @ grad_est_error + grad_vt_Phi_est + grad_vv_Phi @ v_dot + h
        )
        return grad_est_dot

    def _update_law_piecewise_constant_adaptation(
        self,
        grad_est,
        grad_v_Phi,
        grad_vv_Phi,
        T_s,
    ):
        if self.ad_first:
            self.ad_first = False
            return None
        if self.T_s is None:
            const_pre = self._precompute_constant(T_s)
        else:
            const_pre = self.const_pre
        grad_est_error = grad_est - grad_v_Phi
        h = const_pre @ grad_est_error
        torch.linalg.solve(grad_vv_Phi, h, out=self.sigma_hat)
        # sigma_hat = torch.linalg.inv(grad_vv_Phi) @ h
        self.h = h
        # self.sigma_hat = sigma_hat
        return None

    def update(self, t, p, p_dot=None):
        """
        `v`: current (estimate of) optimal solution
        `p`: received parameter (considered as the current parameter)
        `self.update` updates the (estimate of) optimal solution, which can be considered as the (estimate of) optimal solution at the next time step.
        """
        self.baseline_alg._assert_update(t)
        v = self.baseline_alg.v

        if self.grad_est is None:
            self.grad_est = self.baseline_alg._grad_v_Phi_func(v, p)  # initialization

        dt = t - self.baseline_alg.t

        grad_v_Phi = self.baseline_alg._grad_v_Phi_func(v, p)
        if self.grad_est is None:
            self.grad_est = grad_v_Phi
        grad_est = self.grad_est
        grad_vv_Phi = self.baseline_alg._grad_vv_Phi_func(v, p)
        self._update_law_piecewise_constant_adaptation(
            grad_est,
            grad_v_Phi,
            grad_vv_Phi,
            dt,
        )
        if p_dot is None:
            grad_vt_Phi_est = torch.zeros(v.shape)
        else:
            grad_vt_Phi_est = self.baseline_alg._grad_vp_Phi_func(v, p) @ p_dot
        # optimization variable
        v_dot_b = self.baseline_alg._dynamics(grad_v_Phi, grad_vv_Phi, grad_vt_Phi_est)
        v_dot_a = self.v_dot_a

        v_dot = v_dot_b + v_dot_a
        # estimate of gradient
        grad_est_dot = self._update_law_state_predictor(
            grad_est,
            grad_v_Phi,
            grad_vv_Phi,
            grad_vt_Phi_est,
            v_dot,
        )
        sigma_hat = self.sigma_hat

        # update (Euler integration)
        # TODO: gradient predictor requires better integration methods?
        self.baseline_alg.v = v + dt * v_dot
        self.grad_est = grad_est + dt * grad_est_dot

        self.v_dot_a = (
            math.exp(-self.omega * dt) * (v_dot_a + sigma_hat) - sigma_hat
        )  # exact integration
        self.baseline_alg.t = t
        return None

    def get(self):
        return self.baseline_alg.get()
