import numpy as np
import torch
import time

from TVOpt.MPC_L1AO import MPC_L1AO


class PathPlanner:
    def __init__(
            self,
            horizon,
            dt,
            Q_f,
            R,
            obstacle_init,
            obstacle_radius,
            target_init,
            dt_mpc,
            robot_radius_val,
            cost_param_c_val
            ):
        self.mpc_l1ao = MPC_L1AO(
            horizon=horizon,
            dt=dt_mpc,
            Q_f=Q_f,
            R=R,
            obstacles=obstacle_init.copy(),
            cost_param_c=cost_param_c_val,
            robot_radius=robot_radius_val,
            use_analytical_gradients=False,
            omega=10.0,
            A_s=torch.eye(20) * (-0.1)
        )
        self.obstacle_radius = obstacle_radius
        self.robot_radius = robot_radius_val

    def update(self, obstacles, target):
        # The frame is always on robot
        x = np.zeros(2, dtype=np.float32)

        for i, ctr in enumerate(obstacles):
            obstacles[i] = (ctr, self.obstacle_radius)
            # obstacles[i] = (ctr + dt * obstacle_velocities[i], self.obstacle_radius)

        # update internal tensor
        self.mpc_l1ao.obstacles_tensor = torch.tensor(
            [np.concatenate((c, [r])) for c, r in obstacles], dtype=torch.float32
        )

        # solve
        u = self.mpc_l1ao.update_law(x, target)
        # fallback
        if u.shape != x.shape:
            u = np.zeros_like(x)
        
        return u

