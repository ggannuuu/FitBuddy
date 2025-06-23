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
            A_s=torch.eye(10) * (-0.1)
        )
        self.obstacle_radius = obstacle_radius
        self.robot_radius = robot_radius_val

    def update(self, obstacles, target):
        # Validate inputs to prevent nan
        if not np.isfinite(target).all():
            print(f"WARNING: Invalid target values: {target}")
            return np.array([0.0, 0.0])
        
        # Check for reasonable target bounds
        if np.abs(target).max() > 1000:
            print(f"WARNING: Target too far: {target}")
            return np.array([0.0, 0.0])
        
        # Debug: Print shapes and types
        # print(f"obstacles type: {type(obstacles)}")
        # print(f"obstacles shape: {obstacles.shape if hasattr(obstacles, 'shape') else 'No shape attr'}")
        # print(f"obstacles content: {obstacles}")
        # print(f"target type: {type(target)}")
        # print(f"target shape: {target.shape if hasattr(target, 'shape') else 'No shape attr'}")
        # print(f"target content: {target}")
        
        # The frame is always on robot
        x = np.zeros(2, dtype=np.float32)


        # obstacles_np = np.array([
		# 	np.concatenate((c.position()[:2], [r]))
		# 	for c, r in obstacles
		# ])
        
        # self.mpc_l1ao.obstacles_tensor = torch.from_numpy(obstacles_np)

        # Temporarily disable obstacle tensor updates to test
        self.mpc_l1ao.obstacles_tensor = torch.tensor(
            [np.concatenate((c, [r])) for c, r in obstacles], dtype=torch.float32
        )

        # update internal tensor
        # self.mpc_l1ao.obstacles_tensor = torch.tensor(
        #     [np.concatenate((c, [r])) for c, r in obstacles], dtype=torch.float32
        # )

        # solve
        u = self.mpc_l1ao.update_law(x, target)
        # fallback
        if u.shape != x.shape:
            u = np.zeros_like(x)
        
        # Validate output to prevent nan
        if not np.isfinite(u).all():
            print(f"WARNING: MPC returned invalid control: {u}")
            u = np.array([0.0, 0.0])
        
        return u