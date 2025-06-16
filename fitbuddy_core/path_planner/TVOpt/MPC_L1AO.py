import numpy as np
import torch
from TVOpt.L1AO_interactive import L1AO_interactive
from TVOpt.PCIP_interactive import PCIP_interactive  # Baseline algorithm
from TVOpt.TVOptAlgorithm import TVOptAlgorithm
from TVOpt.MPC_PCIP import (
    Phi_cost_function,
    analytical_grad_z_Phi,
    analytical_Hessian_z_Phi,
    analytical_grad_zp_Phi,
)


class MPC_L1AO(TVOptAlgorithm):
    """
    MPC with L1AO_interactive, mirroring MPC_PCIP but using L1AO solver.
    Decision variable z ∈ R^{2N} stacks control inputs u₀…u_{N-1}∈R².
    Dynamics: x_{k+1}=x_k+dt*u_k.
    Cost: control effort + tracking + obstacle avoidance (log-barrier).
    """

    def __init__(
        self,
        horizon,
        dt,
        Q_f=None,
        R=None,
        obstacles=None,
        cost_param_c=1.0,
        robot_radius=0.1,
        use_analytical_gradients=True,
        omega=100.0,
        A_s=None,
        T_s=None,
    ):
        # Store parameters
        self.horizon = horizon
        self.dt = dt
        self.cost_param_c = cost_param_c
        self.robot_radius = robot_radius
        self.use_analytical_gradients = use_analytical_gradients

        # Q_f and R
        n_x = 2
        if Q_f is None:
            Q_f = np.eye(n_x) * 10.0
        if R is None:
            R = np.eye(n_x) * 1.0
        self.Q_f = torch.tensor(Q_f, dtype=torch.float32)
        self.R = torch.tensor(R, dtype=torch.float32)

        # Obstacles: list of (center, radius) → tensor (M,3)
        if not obstacles:
            self.obstacles_tensor = torch.empty((0, 3), dtype=torch.float32)
        else:
            obs_arr = []
            for center, radius in obstacles:
                c = np.array(center, dtype=np.float32)
                obs_arr.append(
                    np.concatenate(
                        (
                            c,
                            [radius],
                        )
                    )
                )
            self.obstacles_tensor = torch.tensor(np.stack(obs_arr), dtype=torch.float32)

        # Baseline PCIP optimizer with same cost setup
        dim_z = 2 * self.horizon
        P = torch.eye(dim_z, dtype=torch.float32)

        def Phi_wrap(z, p):
            return Phi_cost_function(
                z,
                p,
                self.dt,
                self.horizon,
                self.R,
                self.Q_f,
                self.obstacles_tensor,
                self.robot_radius,
                self.cost_param_c,
            )

        self.baseline_optimizer = PCIP_interactive(P=P, Phi_func=Phi_wrap)

        # Override gradients if analytical requested
        if self.use_analytical_gradients:
            self.baseline_optimizer._grad_v_Phi_func = (
                lambda z, p: analytical_grad_z_Phi(
                    z,
                    p,
                    self.dt,
                    self.horizon,
                    self.R,
                    self.Q_f,
                    self.obstacles_tensor,
                    self.robot_radius,
                    self.cost_param_c,
                )
            )
            self.baseline_optimizer._grad_vv_Phi_func = (
                lambda z, p: analytical_Hessian_z_Phi(
                    z,
                    p,
                    self.dt,
                    self.horizon,
                    self.R,
                    self.Q_f,
                    self.obstacles_tensor,
                    self.robot_radius,
                    self.cost_param_c,
                )
            )
            self.baseline_optimizer._grad_vp_Phi_func = (
                lambda z, p: analytical_grad_zp_Phi(
                    z,
                    p,
                    self.dt,
                    self.horizon,
                    self.R,
                    self.Q_f,
                    self.obstacles_tensor,
                    self.robot_radius,
                    self.cost_param_c,
                )
            )

        # L1AO wraps the PCIP baseline
        A_mat = (
            torch.tensor(A_s, dtype=torch.float32)
            if A_s is not None
            else torch.eye(dim_z) * -0.1
        )
        self.optimizer = L1AO_interactive(
            baseline_alg=self.baseline_optimizer,
            omega=omega,
            A_s=A_mat,
            T_s=T_s or dt,
        )

        # Initialize decision variable z0
        z0 = torch.zeros(dim_z, dtype=torch.float32)
        self.optimizer.initialize(z0, 0.0)
        self.p_prev = None
        self.current_time = 0.0

    def update_law(self, x_c, x_d):
        # Convert inputs to torch
        x_c_t = x_c if torch.is_tensor(x_c) else torch.tensor(x_c, dtype=torch.float32)
        x_d_t = x_d if torch.is_tensor(x_d) else torch.tensor(x_d, dtype=torch.float32)
        p = torch.cat([x_c_t, x_d_t])

        # Parameter derivative p_dot
        if self.p_prev is None:
            p_dot = torch.zeros_like(p)
        else:
            p_dot = (p - self.p_prev) / self.dt

        self.p_prev = p.clone()

        # Advance time
        self.current_time += self.dt
        # Run optimizer update
        self.optimizer.update(self.current_time, p, p_dot=p_dot)
        # Get optimal decision variable
        z_opt = self.optimizer.get()
        # Return first control input
        u0 = z_opt[:2]
        return u0.detach().cpu().numpy()

    def dynamics(self, X):
        return {"v": self.update_law(X["x_c"], X["x_d"])}

    def state(self, v):
        return {"v": np.array(v)}
