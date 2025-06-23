import torch
import time


class PCIP_interactive:
    """
    Original equation:
        p(t): (time-dependent) parameter
        v: optimization variable
        Phi(v, p(t)): cost function
        v_dot
        = -inv(grad_vv_Phi(v, p(t))) @ (grad_vt_Phi_est(v, p(t)) + P @ grad_v_Phi(v, p(t)))
        = -inv(grad_vv_Phi(v, p(t))) @ (grad_vp_Phi(v, p(t)) @ p_dot_est + P @ grad_v_Phi(v, p(t)))
    Custom gradient/hessian:
        You can create your own class by overriding the following functions:
            - _grad_v_Phi_func
            - _grad_vv_Phi_func
            - _grad_vp_Phi_func
    """

    def __init__(self, P, Phi_func, eps=None):
        self.P = P
        self.eps = eps
        self.v = None
        self.t = None
        self.v0 = None
        self.t0 = None
        self.Phi_func = Phi_func

    def _grad_v_Phi_func(self, v, p):
        def orig_func(v):
            return self.Phi_func(v, p)

        return torch.func.jacrev(orig_func)(v)

    def _grad_vv_Phi_func(self, v, p):
        def orig_func(v):
            return self._grad_v_Phi_func(v, p)

        return torch.func.jacrev(orig_func)(v)

    def _grad_vp_Phi_func(self, v, p):
        def orig_func(p):
            return self._grad_v_Phi_func(v, p)

        return torch.func.jacrev(orig_func)(p)

    def initialize(self, v0, t0):
        self.v0 = v0
        self.t0 = t0
        self.v = self.v0.detach().clone()
        self.t = self.t0
        return None

    def reset(self):
        self.v = self.v0.detach().clone()
        self.t = self.t0
        return None

    def _dynamics(self, grad_v_Phi, grad_vv_Phi, grad_vt_Phi_est):
        v_dot = torch.empty_like(grad_v_Phi, dtype=self.P.dtype)
        if self.eps is None:
            torch.linalg.solve(
                grad_vv_Phi, -(grad_vt_Phi_est + self.P @ grad_v_Phi), out=v_dot
            )
            # v_dot = -torch.linalg.inv(grad_vv_Phi) @ (
            #     grad_vt_Phi_est + self.P @ grad_v_Phi
            # )
        else:
            norm_grad = torch.linalg.norm(grad_v_Phi)
            torch.linalg.solve(
                grad_vv_Phi,
                -(grad_vt_Phi_est + self.P @ grad_v_Phi / max(norm_grad, self.eps)),
                out=v_dot,
            )
            # v_dot = -torch.linalg.inv(grad_vv_Phi) @ (
            #     grad_vt_Phi_est + self.P @ grad_v_Phi / max(norm_grad, self.eps)
            # )
        return v_dot

    def dynamics(
        self,
        v,
        p,
        p_dot,
    ):

        start_time = time.time()
        grad_v_Phi = self._grad_v_Phi_func(v, p)
        print(f"grad_v_Phi time: {time.time() - start_time:.4f}")
        start_time = time.time()
        grad_vv_Phi = self._grad_vv_Phi_func(v, p)
        print(f"grad_vv_Phi time: {time.time() - start_time:.4f}")
        start_time = time.time()

        if p_dot is None:
            grad_vt_Phi_est = torch.zeros(v.shape, dtype=v.dtype)
        else:
            grad_vt_Phi_est = self._grad_vp_Phi_func(v, p) @ p_dot
        print(f"grad_vt_Phi_est time: {time.time() - start_time:.4f}")
        start_time = time.time()
        v_dot = self._dynamics(grad_v_Phi, grad_vv_Phi, grad_vt_Phi_est)
        print(f"v_dot time: {time.time() - start_time:.4f}")

        return v_dot

    def _assert_update(self, t):
        if self.v0 is None:
            assert False, "initialize first"
        assert t > self.t, "time must be non-decreasing"

    def update(
        self,
        t,
        p,
        p_dot=None,
    ):
        self._assert_update(t)
        v = self.v
        dt = t - self.t
        v_dot = self.dynamics(v, p, p_dot)
        v_next = (
            self.v + dt * v_dot
        )  # Euler integration; can be improved in the future using e.g. Trapezoidal rule
        self.v = v_next
        self.t = t
        return None

    def get(self):
        return self.v
