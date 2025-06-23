import numpy as np
import cvxpy as cp
from .TVOptAlgorithm import TVOptAlgorithm
import time


class MPC(TVOptAlgorithm):
    """
    Tracking MPC algorithm.
    The cost function aims to match the paper's J1+J2+J3:
      1) Control effort (J1): sum_{k=0}^{N-1} u_kᵀ R u_k
      2) Tracking error (J2): sum_{k=1}^{N} (x_k - x_d)ᵀ Q_f (x_k - x_d)
      3) Log-barrier obstacle avoidance (J3): -1/c * sum_{k,i} log(b_i - a_iᵀx_k)
    The optimization is solved over a prediction horizon of N steps, and
    only the first control input is applied (receding horizon).
    """

    def __init__(
        self,
        horizon,
        dt,
        Q_f=np.eye(2) * 2,
        R=np.eye(2),
        obstacles=None,
        cost_param_c=1.0,  # Paper's 'c' or 'alpha' in J3 = -1/c * log(...)
        robot_radius=0.1,  # Paper's r_robot
    ):
        """
        :param horizon: Prediction horizon length (number of steps, N).
        :param dt: Time step.
        :param Q_f: Terminal cost matrix (default: 2x2 identity multiplied by 10).
        :param R: Control input cost matrix (default: 2x2 identity).
        :param obstacles: List of obstacles as (center, radius) tuples.
        :param obstacle_weight: Weight for the obstacle avoidance slack cost.
        :param min_clearance: Minimum required clearance distance from obstacles.
        :param min_distance_weight: Weight for the minimum distance maximization term.
        """
        self.horizon = horizon
        self.dt = dt
        self.Q_f = Q_f if Q_f is not None else np.eye(2)
        self.R = R if R is not None else np.eye(2)
        self.obstacles = obstacles if obstacles is not None else []
        self.cost_param_c = cost_param_c
        self.robot_radius = robot_radius
        self.first_solve = True
        self.epsilon = 1e-6  # For numerical stability in log
        self.setup_optimization(n=2)  # Assuming state dimension n=2

    def compute_states_from_controls(self, u_vars, x0):
        """
        Compute states from controls using dynamics equation.

        :param u_vars: List of control variables [u_0, ..., u_{N-1}]
        :param x0: Initial state
        :return: List of state expressions [x_1, ..., x_N]
        """
        N = self.horizon
        x_vars = []

        # Compute states using dynamics equation
        x_prev = x0
        for k in range(N):
            x_next = x_prev + self.dt * u_vars[k]
            x_vars.append(x_next)
            x_prev = x_next

        return x_vars

    @staticmethod
    def J3_log_barrier_term_cvxpy(
        x_k_var,
        x_current_param,
        robot_radius_val,
        obstacles_list,
        cost_param_c_val,
        epsilon_val,
    ):
        """Computes J3 log-barrier cost for a single state x_k_var."""
        phi_j3_k = 0
        if cost_param_c_val <= 0:
            return phi_j3_k

        for obs_center_val, obs_radius_val in obstacles_list:
            c_i = np.array(obs_center_val)
            R_i = obs_radius_val
            r_R = robot_radius_val

            a_i_expr = c_i - x_current_param
            a_i_norm_sq_expr = cp.sum_squares(a_i_expr) + epsilon_val
            a_i_norm_expr = cp.sqrt(a_i_norm_sq_expr)

            Theta_i_expr = 0.5 - (R_i**2 - r_R**2) / (2 * a_i_norm_sq_expr)

            # Simplified b_i = 0.5*||a_i||^2 - (R_i^2-r_R^2)/2 + va_i^T*x_current - r_R*||a_i||
            b_i_expr = (
                0.5 * a_i_norm_sq_expr
                - (R_i**2 - r_R**2) / 2.0
                + a_i_expr @ x_current_param
                - r_R * a_i_norm_expr
            )

            log_arg = b_i_expr - a_i_expr @ x_k_var + epsilon_val
            phi_j3_k += -(1.0 / cost_param_c_val) * cp.log(log_arg)

        return phi_j3_k

    def setup_optimization(self, n):
        """
        Set up the MPC optimization problem.
        :param n: Dimension of the state (e.g., 2 for 2D position).
        """

        N = self.horizon
        self.x_c_param = cp.Parameter(n, name="x_current")
        self.x_d_param = cp.Parameter(n, name="x_desired")

        self.u_vars = [cp.Variable(n, name=f"u_{k}") for k in range(N)]
        if self.first_solve:
            for u_var in self.u_vars:
                u_var.value = np.zeros(u_var.shape)
            self.first_solve = False

        self.x_vars = self.compute_states_from_controls(self.u_vars, self.x_c_param)

        cost_j1 = sum(cp.quad_form(self.u_vars[k], self.R) for k in range(N))
        cost_j2 = sum(
            cp.quad_form(self.x_vars[k] - self.x_d_param, self.Q_f) for k in range(N)
        )
        cost_j3 = 0
        if self.obstacles and self.cost_param_c > 0:
            processed_obstacles = [
                (np.array(center), radius) for center, radius in self.obstacles
            ]
            for k in range(N):
                cost_j3 += MPC.J3_log_barrier_term_cvxpy(
                    x_k_var=self.x_vars[k],
                    x_current_param=self.x_c_param,
                    robot_radius_val=self.robot_radius,
                    obstacles_list=processed_obstacles,
                    cost_param_c_val=self.cost_param_c,
                    epsilon_val=self.epsilon,
                )

        total_cost = cost_j1 + cost_j2 + cost_j3
        constraints = []  # Add constraints like u_min/max if needed
        self.problem = cp.Problem(cp.Minimize(total_cost), constraints)

    def update_law(self, x_c_val, x_d_val):
        self.x_c_param.value = np.array(x_c_val)
        self.x_d_param.value = np.array(x_d_val)

        # No self.a_list to update as it's removed.
        # a_i and b_i for J3 are now effectively re-derived inside J3_log_barrier_term_cvxpy
        # using x_c_param and current obstacle data for each MPC solve.
        # This is consistent with them being functions of theta_obs(t) which includes x_current.

        solver_options = {
            cp.CLARABEL: {"verbose": False, "tol_gap_abs": 1e-5, "tol_gap_rel": 1e-5},
            cp.ECOS: {"verbose": False, "abstol": 1e-5, "reltol": 1e-5},
            cp.SCS: {
                "verbose": False,
                "eps_abs": 1e-5,
                "eps_rel": 1e-5,
            },  # Added SCS as another option
        }

        # Warm start values for u_vars are retained by cvxpy if not reset
        # For some solvers, explicit warm_start=True is beneficial

        try:
            self.problem.solve(
                solver=cp.CLARABEL, warm_start=True, **solver_options[cp.CLARABEL]
            )
        except (
            cp.SolverError,
            ValueError,
        ):  # ValueError can occur if problem is ill-posed for solver
            # print("CLARABEL failed or problem unsuitable, trying ECOS...")
            try:
                self.problem.solve(
                    solver=cp.ECOS, warm_start=True, **solver_options[cp.ECOS]
                )
            except (cp.SolverError, ValueError):
                # print("ECOS also failed or problem unsuitable, trying SCS...")
                try:
                    self.problem.solve(
                        solver=cp.SCS, warm_start=True, **solver_options[cp.SCS]
                    )
                except (cp.SolverError, ValueError) as e:
                    print(f"All tried solvers (CLARABEL, ECOS, SCS) failed: {e}")
                    u0_fallback = (
                        self.x_d_param.value - self.x_c_param.value
                    ) * self.dt  # Simple fallback
                    # Ensure fallback respects control input dimension if u0_fallback is directly returned
                    if hasattr(self.u_vars[0], "shape"):
                        u0_fallback = np.clip(
                            u0_fallback, -np.inf, np.inf
                        )  # Placeholder for actual limits
                        u0_fallback = u0_fallback[: self.u_vars[0].shape[0]]
                    else:  # Fallback for scalar control
                        u0_fallback = (
                            np.array([u0_fallback])
                            if not isinstance(u0_fallback, np.ndarray)
                            else u0_fallback
                        )

                    return u0_fallback

        if self.problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            # print(f"Warning: Solver finished with status: {self.problem.status}")
            # Fallback if solution is not optimal/acceptable
            u0_fallback = (self.x_d_param.value - self.x_c_param.value) * self.dt
            if hasattr(self.u_vars[0], "shape"):
                u0_fallback = np.clip(u0_fallback, -np.inf, np.inf)
                u0_fallback = u0_fallback[: self.u_vars[0].shape[0]]
            else:
                u0_fallback = (
                    np.array([u0_fallback])
                    if not isinstance(u0_fallback, np.ndarray)
                    else u0_fallback
                )
            return u0_fallback

        u0_optimal = self.u_vars[0].value
        if u0_optimal is None:
            # print("Warning: Solver returned None solution despite OPTIMAL status.")
            u0_fallback = (self.x_d_param.value - self.x_c_param.value) * self.dt
            if hasattr(self.u_vars[0], "shape"):
                u0_fallback = np.clip(u0_fallback, -np.inf, np.inf)
                u0_fallback = u0_fallback[: self.u_vars[0].shape[0]]
            else:
                u0_fallback = (
                    np.array([u0_fallback])
                    if not isinstance(u0_fallback, np.ndarray)
                    else u0_fallback
                )
            return u0_fallback

        return u0_optimal

    def dynamics(self, X):
        """
        Compute the control action based on the current state and target.

        :param X: Dictionary with keys:
            - "x_c": Current state (2D position).
            - "x_d": Target state (2D position).
        :return: Dictionary with key "v" for the computed control (velocity).
        """
        x_c = X["x_c"]
        x_d = X["x_d"]

        return {"v": self.update_law(x_c, x_d)}

    def state(self, v):
        """
        Return the updated state dictionary from the control command.

        :param v: Control command.
        :return: Dictionary representing the state.
        """
        return {"v": np.array(v)}
