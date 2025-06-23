import numpy as np
import torch
from TVOpt.PCIP_interactive import PCIP_interactive
from TVOpt.TVOptAlgorithm import TVOptAlgorithm
import time

# --- Vectorized Helper functions ---


def compute_states_from_controls_vectorized(u_seq, x0_current, dt):
    """
    Vectorized computation of states x_1, ..., x_N.
    """
    dt_u_seq = dt * u_seq
    cumulative_dt_u = torch.cumsum(dt_u_seq, dim=0)
    x_seq = x0_current.unsqueeze(0) + cumulative_dt_u
    return x_seq


def compute_all_a_torch(centers_obs, x_current_state):
    """Vectorized a_vals = centers_obs - x_current_state.unsqueeze(0)"""
    return centers_obs - x_current_state.unsqueeze(0)


def compute_all_b_torch(centers_obs, x_current_state, radii_obs, r_robot_val):
    """Vectorized computation of b_i values for all obstacles."""
    v_i_all = centers_obs - x_current_state.unsqueeze(0)
    v_i_norm_sq_all = torch.sum(v_i_all * v_i_all, dim=1)
    v_i_norm_all = torch.sqrt(v_i_norm_sq_all)

    term1 = 0.5 * v_i_norm_sq_all
    term2 = (radii_obs**2 - r_robot_val**2) / 2.0
    term3 = torch.einsum("oi,i->o", v_i_all, x_current_state)
    term4 = r_robot_val * v_i_norm_all

    return term1 - term2 + term3 - term4


def J3_k_log_barrier_vectorized_torch(
    x_k_state,
    x_current_state,
    robot_radius_val,
    obstacles_tensor,
    cost_param_c_val,
):
    """Vectorized J3 log-barrier cost for a single state x_k over all obstacles."""
    if obstacles_tensor.numel() == 0 or cost_param_c_val <= 0:
        return torch.tensor(0.0, dtype=x_k_state.dtype, device=x_k_state.device)

    centers = obstacles_tensor[:, :2]
    radii = obstacles_tensor[:, 2]

    a_vals = compute_all_a_torch(centers, x_current_state)
    b_vals = compute_all_b_torch(centers, x_current_state, radii, robot_radius_val)

    log_arguments = b_vals - torch.einsum("oi,i->o", a_vals, x_k_state)

    cost_j3_k = -(1.0 / cost_param_c_val) * torch.sum(torch.log(log_arguments))
    return cost_j3_k


def grad_J3_k_log_barrier_vectorized_torch(
    x_k_state,
    x_current_state,
    robot_radius_val,
    obstacles_tensor,
    cost_param_c_val,
):
    """Vectorized gradient of J3_k w.r.t. x_k_state over all obstacles."""
    if obstacles_tensor.numel() == 0 or cost_param_c_val <= 0:
        return torch.zeros_like(x_k_state)

    centers = obstacles_tensor[:, :2]
    radii = obstacles_tensor[:, 2]

    a_vals = compute_all_a_torch(centers, x_current_state)
    b_vals = compute_all_b_torch(centers, x_current_state, radii, robot_radius_val)

    denominators = b_vals - torch.einsum("oi,i->o", a_vals, x_k_state)
    safe_mask = denominators != 0

    grad_contributions = torch.zeros_like(a_vals)
    if torch.any(safe_mask):
        grad_contributions[safe_mask] = torch.einsum(
            "oi,o->oi", a_vals[safe_mask], 1.0 / denominators[safe_mask]
        )

    grad_x_k = (1.0 / cost_param_c_val) * torch.sum(grad_contributions, dim=0)
    return grad_x_k


# --- Main Cost Function (Vectorized J1, J2) ---


def Phi_cost_function(z_flat, p, dt, N, R, Qf, obs_tensor, r_robot, c_param):
    # reshape & split
    n_u = R.size(0)
    n_x = Qf.size(0)
    U = z_flat.view(N, n_u)  # (N, n_u)
    x0, x_des = p[:n_x], p[n_x:]  # each (n_x,)

    # states: (N, n_x)
    X = x0.unsqueeze(0) + dt * U.cumsum(0)

    # J1, J2 as before
    j1 = (U @ R * U).sum()
    E = X - x_des.unsqueeze(0)
    j2 = (E @ Qf * E).sum()

    if obs_tensor.numel() > 0 and c_param > 0:
        # prepare obstacles: (M, n_x)
        centers = obs_tensor[:, :n_x]
        radii = obs_tensor[:, n_x]

        # a: (M,n_x), b: (M,)
        a = centers - x0.unsqueeze(0)
        v = a
        b = (
            0.5 * (v * v).sum(1)
            - (radii**2 - r_robot**2) / 2
            + (v * x0).sum(1)
            - r_robot * torch.norm(v, dim=1)
        )

        # compute all denominators at once: (N, M)
        # X: (N, n_x), a: (M,n_x) -> (N,M) via b - <a, x>
        denom = b.unsqueeze(0) - (X.unsqueeze(1) * a.unsqueeze(0)).sum(-1)
        # single log‐sum
        j3 = -(1.0 / c_param) * torch.log(denom).sum()
    else:
        j3 = X.new_tensor(0.0)

    return j1 + j2 + j3


# --- Analytical Gradients (Partially Vectorized) ---


def analytical_grad_z_Phi(z_flat, p, dt, N, R, Qf, obs_tensor, r_robot, c_param):
    # first compute state gradient ∂Φ/∂X in one shot
    n_u, n_x = R.size(0), Qf.size(0)
    U = z_flat.view(N, n_u)
    x0, x_des = p[:n_x], p[n_x:]
    X = x0.unsqueeze(0) + dt * U.cumsum(0)
    # ∂J2/∂X: (N,n_x)
    grad_J2 = 2 * (X - x_des.unsqueeze(0)) @ Qf
    # ∂J3/∂X: similar vectorization
    if obs_tensor.numel() > 0 and c_param > 0:
        centers = obs_tensor[:, :n_x]
        radii = obs_tensor[:, n_x]
        a = centers - x0.unsqueeze(0)  # (M,n_x)
        v = a
        b = (
            0.5 * (v * v).sum(1)
            - (radii**2 - r_robot**2) / 2
            + (v * x0).sum(1)
            - r_robot * torch.norm(v, dim=1)
        )
        denom = b.unsqueeze(0) - (X.unsqueeze(1) * a.unsqueeze(0)).sum(-1)
        # broadcast division: (N,M,n_x)
        grad_J3 = (a.unsqueeze(0) / denom.unsqueeze(2)).sum(1) * (1.0 / c_param)
    else:
        grad_J3 = X.new_zeros(N, n_x)

    # backward‐prop through X= x0 + dt cumsum(U):
    # ∂Φ/∂U = dt * [ grad_X[0] + (grad_X[1] + grad_X[0]) + ... + (grad_X[N-1]+...+grad_X[0]) ]
    G = grad_J2 + grad_J3  # (N,n_x)
    # cumsum from bottom
    G_rev = torch.flip(torch.cumsum(torch.flip(G, [0]), 0), [0])  # (N,n_x)
    grad_J1 = 2 * U @ R.T  # (N,n_u)
    # now map ∂Φ/∂X into control space: dt * G_rev, then reshape
    grad_tot = grad_J1 + dt * G_rev
    return grad_tot.reshape(-1)


# --- Hessian and Mixed Derivatives (Vectorized inner obstacle sums) ---


def analytical_Hessian_z_Phi(
    z_controls_flat,
    p_params,
    dt_val,
    horizon_N,
    R_matrix,
    Q_f_matrix,
    obstacles_tensor_val,
    robot_r_val,
    cost_param_c_val,
):
    n_u = R_matrix.shape[0]
    n_x = Q_f_matrix.shape[0]
    dim_z = horizon_N * n_u
    Hessian = torch.zeros(
        (dim_z, dim_z), dtype=z_controls_flat.dtype, device=z_controls_flat.device
    )

    # J1 Hessian
    for k in range(horizon_N):
        idx = slice(n_u * k, n_u * (k + 1))
        Hessian[idx, idx] += 2 * R_matrix

    # J2 Hessian
    for l_idx in range(horizon_N):
        for m_idx in range(horizon_N):
            idx_l = slice(n_u * l_idx, n_u * (l_idx + 1))
            idx_m = slice(n_u * m_idx, n_u * (m_idx + 1))
            common = horizon_N - max(l_idx, m_idx)
            if common > 0:
                Hessian[idx_l, idx_m] += common * 2 * (dt_val**2) * Q_f_matrix

    # J3 Hessian
    if obstacles_tensor_val.numel() > 0 and cost_param_c_val > 0:
        u_seq = z_controls_flat.reshape(horizon_N, n_u)
        x_current = p_params[:n_x]
        x_seq = compute_states_from_controls_vectorized(u_seq, x_current, dt_val)
        centers = obstacles_tensor_val[:, :2]

        for s_idx in range(horizon_N):
            x_state = x_seq[s_idx]
            a_vals = compute_all_a_torch(centers, x_current)
            b_vals = compute_all_b_torch(
                centers, x_current, obstacles_tensor_val[:, 2], robot_r_val
            )
            denom_sq = (b_vals - torch.einsum("oi,i->o", a_vals, x_state)) ** 2
            mask = denom_sq != 0
            H_terms = torch.zeros(
                (centers.shape[0], n_x, n_x),
                dtype=z_controls_flat.dtype,
                device=z_controls_flat.device,
            )
            if torch.any(mask):
                safe = torch.einsum("oi,oj->oij", a_vals[mask], a_vals[mask])
                H_terms[mask] = safe / denom_sq[mask].view(-1, 1, 1)
            H_xx = (1.0 / cost_param_c_val) * torch.sum(H_terms, dim=0)
            for l_idx in range(s_idx + 1):
                for m_idx in range(s_idx + 1):
                    idx_l = slice(n_u * l_idx, n_u * (l_idx + 1))
                    idx_m = slice(n_u * m_idx, n_u * (m_idx + 1))
                    Hessian[idx_l, idx_m] += (dt_val**2) * H_xx

    return Hessian


def analytical_grad_zp_Phi(z_flat, p, dt, N, R, Qf, obs_tensor, r_robot, c_param):
    """
    Analytical Jacobian of ∇_zΦ w.r.t. parameters p (x0 and x_des),
    implemented via torch.func.jacrev.
    """
    # First, we need a function (analytical_grad_z_Phi) to compute ∇_zΦ
    # Then, we compute the Jacobian (∂/∂p) with respect to p for that function
    jac_fn = torch.func.jacrev(
        lambda zz, pp: analytical_grad_z_Phi(
            zz, pp, dt, N, R, Qf, obs_tensor, r_robot, c_param
        ),
        argnums=1,  # Differentiate with respect to the second argument (p)
    )
    # jac will be a tensor of size (dim_z, dim_p)
    return jac_fn(z_flat, p)


# MPC_PCIP Class
class MPC_PCIP(TVOptAlgorithm):
    def __init__(
        self,
        horizon,
        dt,
        Q_f=None,
        R=None,
        obstacles=None,
        cost_param_c=1.0,
        robot_radius=0.1,
        use_analytical_gradients=False,
    ):
        self.horizon = horizon
        self.dt = dt
        self.robot_radius = robot_radius
        self.cost_param_c = cost_param_c
        self.n_x = 2
        self.n_u = 2

        default_Q_f = np.eye(self.n_x) * 5.0
        default_R = np.eye(self.n_u) * 1.0
        self.Q_f_matrix = torch.tensor(
            Q_f if Q_f is not None else default_Q_f, dtype=torch.float32
        )
        self.R_matrix = torch.tensor(
            R if R is not None else default_R, dtype=torch.float32
        )

        if obstacles is None or len(obstacles) == 0:
            self.obstacles_tensor = torch.empty((0, 3), dtype=torch.float32)
        else:
            obs_list = [
                np.concatenate(
                    (np.array(c, dtype=np.float32), np.array([r], dtype=np.float32))
                )
                for c, r in obstacles
            ]
            self.obstacles_tensor = torch.tensor(
                np.array(obs_list), dtype=torch.float32
            )

        P_mat = torch.eye(self.n_u * self.horizon, dtype=torch.float32)

        def Phi_wrapper(z, p):
            return Phi_cost_function(
                z,
                p,
                self.dt,
                self.horizon,
                self.R_matrix,
                self.Q_f_matrix,
                self.obstacles_tensor,
                self.robot_radius,
                self.cost_param_c,
            )

        self.optimizer = PCIP_interactive(P=P_mat, Phi_func=Phi_wrapper)

        if use_analytical_gradients:
            self.optimizer._grad_v_Phi_func = lambda z, p: analytical_grad_z_Phi(
                z,
                p,
                self.dt,
                self.horizon,
                self.R_matrix,
                self.Q_f_matrix,
                self.obstacles_tensor,
                self.robot_radius,
                self.cost_param_c,
            )
            self.optimizer._grad_vv_Phi_func = lambda z, p: analytical_Hessian_z_Phi(
                z,
                p,
                self.dt,
                self.horizon,
                self.R_matrix,
                self.Q_f_matrix,
                self.obstacles_tensor,
                self.robot_radius,
                self.cost_param_c,
            )
            self.optimizer._grad_vp_Phi_func = lambda z, p: analytical_grad_zp_Phi(
                z,
                p,
                self.dt,
                self.horizon,
                self.R_matrix,
                self.Q_f_matrix,
                self.obstacles_tensor,
                self.robot_radius,
                self.cost_param_c,
            )
        self.z0 = torch.zeros(self.n_u * self.horizon, dtype=torch.float32)
        self.optimizer.initialize(self.z0, 0.0)
        self.p_prev = None

    def update_law(self, x_c_val, x_d_val):
        x_c_tensor = (
            torch.tensor(x_c_val, dtype=torch.float32)
            if not torch.is_tensor(x_c_val)
            else x_c_val.float()
        )
        x_d_tensor = (
            torch.tensor(x_d_val, dtype=torch.float32)
            if not torch.is_tensor(x_d_val)
            else x_d_val.float()
        )
        p_current = torch.cat([x_c_tensor, x_d_tensor])

        p_dot = torch.zeros_like(p_current)
        if self.p_prev is not None and torch.norm(p_current - self.p_prev) > 1e-6:
            p_dot = (p_current - self.p_prev) / self.dt
        self.p_prev = p_current.clone()

        current_t = getattr(self.optimizer, "t", 0.0)
        self.optimizer.update(current_t + self.dt, p_current, p_dot=p_dot)

        z_opt = self.optimizer.v
        return z_opt[: self.n_u].detach().numpy()

    def dynamics(self, X):
        return {"v": self.update_law(X["x_c"], X["x_d"])}

    def state(self, v):
        return {"v": np.array(v)}
