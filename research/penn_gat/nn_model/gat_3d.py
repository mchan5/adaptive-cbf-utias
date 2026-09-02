"""
gat_3d.py — 3D Graph Attention Network for the quadrotor PENN+GAT pipeline.

Differences from the original gat.py (bicycle model):
  - robot input: [x, y, z, vx, vy, vz]        (was [x, y, vx, vy])
  - obstacle input: [ox, oy, oz, r, vx, vy, vz] (was [ox, oy, r, vx, vy, ...])
  - goal input: [gx, gy, gz]                    (was [gx, gy])
  - node features: one-hot(3) + prox(1) + vel(3) = 7  (was 6)
  - edge attributes: [dx, dy, dz, dist, dvx, dvy, dvz] = 7  (was 5)
  - zij_dim: 7+7+7 = 21  (was 17)
  - psi4 input: 16 + gamma_dim  (was hardcoded 18)
  - gamma_dim: 1 for the drone (alpha_obs only)
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_sum, scatter_softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class GATModule3D(nn.Module):

    def __init__(self, robot_radius=0.25, gamma_dim=1, lr=0.001,
                 num_epochs=50, batch_size=8, device='cpu'):
        super().__init__()
        self.robot_radius = robot_radius
        self.gamma_dim = gamma_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        self.gat = self.GATGraphNetwork3D(gamma_dim=gamma_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.gat.parameters(), lr=self.lr)

    class GATGraphNetwork3D(nn.Module):
        """
        3D attention graph network.

        Node features  : [one-hot(3), proximity(1), vx, vy, vz] = 7
        Edge attributes: [dx, dy, dz, distance, dvx, dvy, dvz]  = 7
        zij_dim        : 7 + 7 + 7 = 21
        Output         : [low_speed_time, risk_horizon]  (shape [B, 2]) --
                         both now receding-horizon quantities (Stage 2,
                         2026-08-18 barrier-label-collapse plan) rather than
                         whole-episode-from-start ones: low_speed_time
                         replaced t_goal (which itself had replaced
                         deadlock_time) and risk_horizon is the same
                         violation-integral accumulation windowed to a
                         forward horizon instead of the full episode. This
                         class is label-agnostic and needs no code change
                         either way -- see drone_data_generation.py's
                         simulate_horizon() for the actual label semantics.
        """

        def __init__(self, gamma_dim=1):
            super().__init__()
            self.gamma_dim = gamma_dim
            self.zij_dim = 21  # 7 node + 7 node + 7 edge

            self.psi1 = nn.Sequential(
                nn.Linear(self.zij_dim, 64), nn.ReLU(), nn.Linear(64, 16)
            )
            self.psi2 = nn.Sequential(
                nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
            )
            self.psi3 = nn.Sequential(
                nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 16)
            )
            self.psi4 = nn.Sequential(
                nn.Linear(16 + gamma_dim, 64), nn.ReLU(), nn.Linear(64, 2)
            )

        def forward(self, x, edge_index, edge_attr, batch, gammas=None):
            device = next(self.parameters()).device
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch = batch.to(device)

            src, dst = edge_index
            zij = torch.cat([x[src], x[dst], edge_attr], dim=1).to(device)

            q_ij = self.psi1(zij)
            attn = scatter_softmax(self.psi2(q_ij).squeeze(-1), src)
            msg = self.psi3(q_ij)
            node_emb = scatter_sum(attn.unsqueeze(-1) * msg, src, dim=0)

            first_nodes = torch.cat((
                torch.tensor([0], device=device),
                (torch.where(torch.diff(batch))[0] + 1).to(device),
            ))
            robot_q = node_emb[first_nodes]

            gammas = gammas.to(device)
            return self.psi4(torch.cat([robot_q, gammas], dim=1))

        def extract_robot_embedding(self, x, edge_index, edge_attr, batch):
            """Return the 16-D robot embedding (psi1–psi3 only, no psi4)."""
            device = next(self.parameters()).device
            x = x.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            batch = batch.to(device)

            src, dst = edge_index
            zij = torch.cat([x[src], x[dst], edge_attr], dim=1).to(device)

            q_ij = self.psi1(zij)
            attn = scatter_softmax(self.psi2(q_ij).squeeze(-1), src)
            msg = self.psi3(q_ij)
            node_emb = scatter_sum(attn.unsqueeze(-1) * msg, src, dim=0)

            first_nodes = torch.cat((
                torch.tensor([0], device=device),
                (torch.where(torch.diff(batch))[0] + 1).to(device),
            ))
            return node_emb[first_nodes]  # [B, 16]

        def extract_robot_velocity(self, x, batch):
            """Return the robot node's raw [vx, vy, vz] (node feature columns
            4:7 -- see create_graph's node_features layout: one-hot(3) +
            prox(1) + vel(3)).

            Added 2026-08-19 (Phase B4 follow-up, barrier-label-collapse
            plan): the ensemble PENN head (nn_gat_iccbf_predict.py) only ever
            saw velocity indirectly, filtered through extract_robot_
            embedding's attention pooling over the whole graph -- there was
            no direct path from velocity to the final prediction. Diagnosed
            after two retrains: the risk head (same architecture, same
            training run) correctly tracked ground truth across all tested
            query velocities, while the performance head did not -- ground
            truth for performance genuinely REVERSES SIGN with velocity
            (weak correction is best from rest, strong correction is best
            already at speed; diagnose_perf_head_vel.py /
            diagnose_risk_head_vel.py), and the pooled embedding apparently
            wasn't preserving enough of that signal for the small ensemble
            MLP to recover it. Giving the head a direct, unobscured velocity
            input removes the need for that reconstruction.
            """
            device = next(self.parameters()).device
            x = x.to(device)
            batch = batch.to(device)
            first_nodes = torch.cat((
                torch.tensor([0], device=device),
                (torch.where(torch.diff(batch))[0] + 1).to(device),
            ))
            return x[first_nodes, 4:7]  # [B, 3]

    # ── Graph construction ────────────────────────────────────────────────────

    @staticmethod
    def _canonicalize_frame(robot, obstacles, goal):
        """Rotate the whole scene about z so (goal - robot_pos) points along
        +x, before any feature is derived from it.

        Added 2026-08-20 (Stage 2, barrier-label-collapse plan's scene-level
        reframe). Every edge attribute already includes raw [dx, dy, dz] --
        only the derived `dist` component is actually rotation-invariant --
        and extract_robot_velocity() reads a raw world-frame [vx, vy, vz]
        node feature with no rotation applied at all. Random training scenes
        are generated with the goal at x in [2,7] from a robot starting near
        the origin, so training data is *approximately* +x-forward by
        construction, but the deployed course flies +y: a scene rotated 90
        degrees is physically identical but was measured to change the
        selected gamma on 59% of scenes (mean |delta gamma| 0.78, checked
        against the deployed checkpoint) and to roughly halve the aleatoric
        safety gate's rejection rate, purely from heading.

        Canonicalizing here, rather than only fixing the direct-velocity
        input, closes the gap at its source: both drone_data_generation.py's
        labelling and cbf_obstacle_node.py's inference call this same
        function, so training and deployment become frame-consistent by
        construction instead of by the coincidence of how scenes happen to
        be sampled. Only yaw (rotation about z) is removed -- z is gravity-
        aligned and left untouched, matching the small (12-14%) share of
        real flight speed ever carried by vz.
        """
        robot = list(robot)
        goal = list(goal)
        rx, ry = goal[0] - robot[0], goal[1] - robot[1]
        yaw = np.arctan2(ry, rx)
        c, s = np.cos(-yaw), np.sin(-yaw)

        def rot_xy(vx, vy):
            return c * vx - s * vy, s * vx + c * vy

        def rot_pos(p):
            x, y = rot_xy(p[0], p[1])
            return [x, y, p[2]]

        def rot_vel(v):
            x, y = rot_xy(v[0], v[1])
            return [x, y, v[2]]

        robot_r = rot_pos(robot[:3]) + rot_vel(robot[3:6])
        goal_r = rot_pos(goal[:3])
        obstacles_r = [rot_pos(list(o[:3])) + [o[3]] + rot_vel(list(o[4:7]))
                      for o in obstacles]
        return robot_r, obstacles_r, goal_r

    def create_graph(self, robot, obstacles, goal, deadlock=0.0, risk=0.0):
        """
        Build a PyG Data object for one 3D scene.

        Args:
            robot    : [x, y, z, vx, vy, vz]
            obstacles: list of [ox, oy, oz, radius, vx, vy, vz]
            goal     : [gx, gy, gz]
            deadlock : float label (low_speed_time over a receding horizon as
                       of Stage 2, 2026-08-18, despite the arg name -- kwarg
                       kept for minimal API churn across the label swaps)
            risk     : float label (risk_horizon: violation integral over
                       the same receding horizon, as of Stage 2 2026-08-18)
        """
        robot, obstacles, goal = self._canonicalize_frame(robot, obstacles, goal)
        n_obs = len(obstacles)
        num_nodes = n_obs + 2  # robot + obstacles + goal

        # One-hot node types [N, 3]
        node_features = torch.tensor(
            [[1, 0, 0]] + [[0, 1, 0]] * n_obs + [[0, 0, 1]],
            dtype=torch.float,
        )

        # 3-D positions [N, 3]
        robot_pos = torch.tensor(robot[:3], dtype=torch.float).unsqueeze(0)
        obs_pos = (
            torch.from_numpy(np.array([o[:3] for o in obstacles], dtype=np.float32))
            if n_obs > 0 else torch.empty(0, 3)
        )
        goal_pos = torch.tensor(goal[:3], dtype=torch.float).unsqueeze(0)
        all_pos = torch.cat([robot_pos, obs_pos, goal_pos], dim=0)  # [N, 3]

        # Radii [N]
        radii = torch.tensor(
            [self.robot_radius] + [o[3] for o in obstacles] + [0.0],
            dtype=torch.float,
        )

        # Fully connected edges (all i→j, i≠j)
        edge_idx = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
        edge_index = torch.tensor(edge_idx, dtype=torch.long).t()

        # Relative positions [N, N, 3]
        rel_pos = all_pos.unsqueeze(0) - all_pos.unsqueeze(1)  # [N, N, 3]

        # Edge-to-edge distances (subtracting radii) [N, N, 1]
        dist_raw = torch.norm(rel_pos, dim=2)                          # [N, N]
        radii_mat = radii.unsqueeze(0) + radii.unsqueeze(1)            # [N, N]
        dist = torch.clamp(dist_raw - radii_mat, min=0.0).unsqueeze(-1)  # [N, N, 1]

        # Per-node proximity: distance to nearest obstacle [N, 1]
        dist_2d = dist[:, :, 0]
        if n_obs > 0:
            obs_cols = list(range(1, n_obs + 1))
            robot_prox = dist_2d[0, obs_cols].min().unsqueeze(0)
            obs_prox   = dist_2d[obs_cols, 0]
            goal_prox  = dist_2d[n_obs + 1, obs_cols].min().unsqueeze(0)
        else:
            robot_prox = torch.tensor([0.0])
            obs_prox   = torch.empty(0)
            goal_prox  = torch.tensor([0.0])
        node_prox = torch.cat([robot_prox, obs_prox, goal_prox]).unsqueeze(1)
        node_features = torch.cat([node_features, node_prox], dim=1)  # [N, 4]

        # 3-D velocities [N, 3]
        robot_vel = torch.tensor(robot[3:6], dtype=torch.float).unsqueeze(0)
        obs_vel = (
            torch.from_numpy(np.array([o[4:7] for o in obstacles], dtype=np.float32))
            if n_obs > 0 else torch.zeros(0, 3)
        )
        goal_vel = torch.zeros(1, 3)
        all_vels = torch.cat([robot_vel, obs_vel, goal_vel], dim=0)  # [N, 3]

        # Relative velocities [N, N, 3]
        rel_vel = all_vels.unsqueeze(0) - all_vels.unsqueeze(1)  # [N, N, 3]

        # Edge attributes: [dx, dy, dz, dist, dvx, dvy, dvz] = 7
        edge_features = torch.cat([rel_pos, dist, rel_vel], dim=2).view(-1, 7)
        src_lin = edge_index[0] * num_nodes + edge_index[1]
        edge_attr = edge_features[src_lin]

        # Final node features: [one-hot(3), prox(1), vx, vy, vz] = 7
        node_features = torch.cat([node_features, all_vels], dim=1)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.y = torch.tensor([[deadlock, risk]], dtype=torch.float)
        return data
