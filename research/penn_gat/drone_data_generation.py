"""Generates PENN+GAT training data for the quadrotor CBF using a lightweight Python drone
integrator. No ROS2, no Gazebo, no safe_control dependency."""

import os
import sys
import numpy as np
import pickle
import tqdm
import torch
from torch.multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nn_model'))
from gat_3d import GATModule3D

DRONE_RADIUS  = 0.25   # x500 body half-width
SAFETY_MARGIN = 0.5    # added to obstacle radius at inference time (matches obstacle_cbf_node.py)
DEPLOY_GAMMA_RANGE = (1.3, 8.0)
# penn_gamma_min/max at inference (updated 2026-08-23, range-widening plan: the closed-loop
# controller was measured safe (0 hard collisions, 300 test scenes) to gamma=8.0, and the best …
GAMMA_RANGE   = (0.2, 9.0)
UNSAFE_PROB   = 0.20   # fraction of episodes run WITHOUT CBF so risk > 0 labels appear

# ── Sensing model — mirrors lidar_obstacle_publisher_node.py's defaults ────── (2026-08-20,
# recursive-feasibility redesign, Phase 0): every simulated actor -- training-data generation, the …
SENSOR_MAX_RANGE = 8.0
SENSOR_Z_MIN = 0.1
SENSOR_Z_MAX = 2.5


def visible_obstacles(pos, obstacles):
    """Filter `obstacles` (list of (center, radius)) down to what a perception pipeline with this
    sensing envelope could actually report from `pos` right now (distance measured to the …"""
    out = []
    for c, r in obstacles:
        if not (SENSOR_Z_MIN <= c[2] <= SENSOR_Z_MAX):
            continue
        dist_to_surface = float(np.linalg.norm(np.asarray(pos) - np.asarray(c))) - r
        if dist_to_surface <= SENSOR_MAX_RANGE:
            out.append((c, r))
    return out


# ── CBF helpers — mirror of obstacle_cbf_node.py ─────────────────────────────

def cbf_obstacle(pos, v, center, radius, alpha):
    n = pos - center
    n_sq = float(np.dot(n, n))
    if n_sq < 1e-9:
        return v
    h = n_sq - radius ** 2
    lf_h = 2.0 * float(np.dot(n, v))
    slack = lf_h + alpha * h
    if slack >= 0:
        return v
    return v + (-slack / (2.0 * n_sq)) * n


def cbf_obstacle_accel(pos, vel, a_nom, center, radius, gamma):
    """Relative-degree-2 HOCBF obstacle avoidance, applied directly to acceleration (the
    vehicle's real control input) instead of to the P-tracked target velocity."""
    n = pos - center
    n_sq = float(np.dot(n, n))
    if n_sq < 1e-6:
        return a_nom, False
    h = n_sq - radius ** 2
    hdot = 2.0 * float(np.dot(n, vel))
    g = 2.0 * n  # d(hddot)/d(a)
    rhs = -2.0 * float(np.dot(vel, vel)) - 2.0 * gamma * hdot - gamma ** 2 * h
    slack = float(np.dot(g, a_nom)) - rhs
    if slack >= 0:
        return a_nom, True
    g_sq = float(np.dot(g, g))
    if g_sq < 1e-9:
        return a_nom, False
    correction = ((-slack) / g_sq) * g
    feasible = h > -radius ** 2
    return a_nom + correction, feasible


def cbf_boundary(pos, v, bnd, alpha):
    x_min, x_max, y_min, y_max, z_min, z_max = bnd
    x, y, z = pos
    vx = float(np.clip(v[0], -alpha * (x - x_min), alpha * (x_max - x)))
    vy = float(np.clip(v[1], -alpha * (y - y_min), alpha * (y_max - y)))
    vz = float(np.clip(v[2], -alpha * (z - z_min), alpha * (z_max - z)))
    return np.array([vx, vy, vz])


# ── Shared per-step dynamics ──────────────────────────────────────────────────

def step_dynamics(pos, vel, goal, obstacles, alpha_obs, boundary, alpha_bnd,
                   v_max, k_p, k_v, a_max, apply_cbf):
    """One 'dt-normalized' control step: goal+boundary tracking in velocity space, obstacle
    avoidance in acceleration space, a_max clip."""
    err = goal - pos
    v_nom = k_p * err
    speed = float(np.linalg.norm(v_nom))
    if speed > v_max:
        v_nom = v_nom / speed * v_max

    v_safe = cbf_boundary(pos, v_nom, boundary, alpha_bnd)

    # Ground truth: did we actually get close to something, full obstacle set, regardless of what's
    # currently sensed.
    h_step = np.inf
    for center, radius in obstacles:
        n = pos - center
        h = float(np.dot(n, n)) - radius ** 2
        if h < h_step:
            h_step = h

    a_safe = k_v * (v_safe - vel)

    if apply_cbf:
        # Avoidance can only react to what's currently sensed -- matches obstacle_cbf_node.py, which
        # only ever has whatever the (filtered) /arc/obstacles topic last published in …
        for center, radius in visible_obstacles(pos, obstacles):
            a_safe, _ = cbf_obstacle_accel(pos, vel, a_safe, center, radius, alpha_obs)

    a_norm = float(np.linalg.norm(a_safe))
    clipped = a_norm > a_max
    if clipped:
        a_safe = a_safe / a_norm * a_max

    return a_safe, h_step, clipped


# ── Episode simulator ─────────────────────────────────────────────────────────

def simulate_episode(
    start, goal, obstacles, alpha_obs,
    boundary=(-1.0, 8.0, -3.0, 3.0, 0.3, 3.0),
    alpha_bnd=0.5, v_max=3.0, k_p=0.5, k_v=2.0, a_max=5.0,
    dt=0.05, max_steps=400,
    apply_cbf=True,
):
    """Run one drone episode with the given CBF parameters."""
    pos = start.copy()
    vel = np.zeros(3)
    min_h = np.inf
    viol_integral = 0.0
    n_steps = 0
    n_clipped = 0

    for step in range(max_steps):
        err = goal - pos
        if float(np.linalg.norm(err)) < 0.3:
            t_goal = step * dt
            frac_clipped = (n_clipped / n_steps) if n_steps else 0.0
            return (min_h if np.isfinite(min_h) else 0.0, viol_integral, t_goal, True, frac_clipped)

        a_safe, h_step, clipped = step_dynamics(
            pos, vel, goal, obstacles, alpha_obs, boundary, alpha_bnd,
            v_max, k_p, k_v, a_max, apply_cbf,
        )

        if h_step < min_h:
            min_h = h_step
        if h_step < 0.0:
            viol_integral += (-h_step) * dt

        n_steps += 1
        if clipped:
            n_clipped += 1

        vel = vel + a_safe * dt
        pos = pos + vel * dt

    frac_clipped = (n_clipped / n_steps) if n_steps else 0.0
    return (min_h if np.isfinite(min_h) else 0.0, viol_integral, max_steps * dt, False, frac_clipped)


# ── Receding-horizon simulator ────────────────────────────────────────────────


def simulate_horizon(
    pos0, vel0, goal, obstacles, alpha_obs,
    boundary=(-1.0, 8.0, -3.0, 3.0, 0.3, 3.0),
    alpha_bnd=0.5, v_max=3.0, k_p=0.5, k_v=2.0, a_max=5.0,
    dt=0.05, horizon_s=4.0,
    apply_cbf=True,
):
    """Simulate forward from an arbitrary (pos0, vel0) state for horizon_s seconds under a fixed
    candidate gamma, and return receding-horizon performance/risk metrics."""
    pos = pos0.copy()
    vel = vel0.copy()
    dist0 = float(np.linalg.norm(goal - pos))
    risk_horizon = 0.0
    min_h_horizon = np.inf
    n_steps = 0
    n_clipped = 0
    n_horizon_steps = max(1, int(round(horizon_s / dt)))
    dist_final = dist0

    for _ in range(n_horizon_steps):
        err = goal - pos
        dist = float(np.linalg.norm(err))
        dist_final = dist
        if dist < 0.3:
            # Reached goal inside the horizon: remaining time counts as
            # neither risky nor a progress deficit (nothing left to do).
            break

        a_safe, h_step, clipped = step_dynamics(
            pos, vel, goal, obstacles, alpha_obs, boundary, alpha_bnd,
            v_max, k_p, k_v, a_max, apply_cbf,
        )

        if h_step < 0.0:
            risk_horizon += (-h_step) * dt
        if h_step < min_h_horizon:
            min_h_horizon = h_step

        n_steps += 1
        if clipped:
            n_clipped += 1

        vel = vel + a_safe * dt
        pos = pos + vel * dt
        dist_final = float(np.linalg.norm(goal - pos))

    ideal_progress = min(v_max * horizon_s, dist0)
    actual_progress = dist0 - dist_final
    progress_deficit = max(0.0, ideal_progress - actual_progress)

    frac_clipped = (n_clipped / n_steps) if n_steps else 0.0
    min_h_horizon = min_h_horizon if np.isfinite(min_h_horizon) else 0.0
    return risk_horizon, progress_deficit, frac_clipped, min_h_horizon


# ── Scene randomization ───────────────────────────────────────────────────────

def random_scene(n_obs_range=(1, 9), tight_prob=0.7, near_goal_prob=0.15):
    """Sample a random 3D scene: start, goal, and N sphere obstacles."""
    # Goal always 2–7 m from origin
    goal = np.array([
        np.random.uniform(2.0, 7.0),
        np.random.uniform(-1.5, 1.5),
        np.random.uniform(1.0, 2.0),
    ])

    near_goal_flag = np.random.random() < near_goal_prob
    if near_goal_flag:
        # Near-goal scene: robot is 0.5–2.0 m from the goal in a random direction.
        # Covers the waypoint-holding OOD case (robot reached goal, obstacle nearby).
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        offset_len = np.random.uniform(0.5, 2.0)
        start = goal + direction * offset_len
        start = np.clip(start, [-1.0, -3.0, 0.3], [8.0, 3.0, 3.0])
    else:
        # Normal scene: start near origin at flight altitude
        start = np.array([
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(1.2, 1.8),
        ])

    n_obs = np.random.randint(n_obs_range[0], n_obs_range[1])
    obstacles = []
    path = goal - start
    path_len = float(np.linalg.norm(path))
    if path_len < 1e-6:
        path_len = 1.0

    for _ in range(500):
        if len(obstacles) >= n_obs:
            break

        radius = np.random.uniform(0.2, 0.5)
        eff_r  = radius + SAFETY_MARGIN
        margin = DRONE_RADIUS + eff_r

        if near_goal_flag:
            # For near-goal scenes the path is short (0.5–2 m) so path-based placement rarely clears
            # the start/goal margin.
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            dist = np.random.uniform(margin + 0.05, margin + 1.5)
            center = start + direction * dist
            center[2] = np.clip(center[2], 0.3, 3.0)
        elif np.random.random() < tight_prob:
            t = np.random.uniform(0.25, 0.85)
            # Tighter lateral spread than before (was ±0.9/±0.5) to stress CBF more
            lateral = np.array([
                np.random.uniform(-0.6, 0.6),
                np.random.uniform(-0.6, 0.6),
                np.random.uniform(-0.3, 0.3),
            ])
            center = start + t * path + lateral
        else:
            center = np.array([
                np.random.uniform(0.3, max(goal[0] - 0.3, 0.5)),
                np.random.uniform(-2.5, 2.5),
                np.random.uniform(0.5, 2.5),
            ])

        if (np.linalg.norm(center - start) < margin or
                np.linalg.norm(center - goal) < margin):
            continue

        if all(np.linalg.norm(center - c) >= r + radius + DRONE_RADIUS * 2
               for c, r in obstacles):
            obstacles.append((center, radius))

    return start, goal, obstacles


def random_gate_scene(n_extra_range=(0, 4)):
    """Sample a scene with a deliberate two-obstacle "gate" directly on the path, plus optional
    extra scattered obstacles elsewhere. Added 2026-08-19 (Stage 2 follow-up, barrier-label- …"""
    goal = np.array([
        np.random.uniform(4.0, 7.0),
        np.random.uniform(-1.0, 1.0),
        np.random.uniform(1.0, 2.0),
    ])
    start = np.array([
        np.random.uniform(-0.2, 0.2),
        np.random.uniform(-0.2, 0.2),
        np.random.uniform(1.2, 1.8),
    ])

    gate_x = np.random.uniform(2.0, goal[0] - 1.0)
    gate_y = np.random.uniform(-0.3, 0.3)  # gate center can be off-axis
    gate_z = np.random.uniform(1.0, 2.0)
    obs_radius = np.random.uniform(0.2, 0.4)
    # Half-width sampled so the *effective* (SAFETY_MARGIN-inclusive) gap
    # spans roughly [-0.8, +1.5] m: solidly impossible through solidly open.
    eff_r = obs_radius + SAFETY_MARGIN
    gate_half_width = np.random.uniform(eff_r - 0.4, eff_r + 0.75)

    obstacles = [
        (np.array([gate_x, gate_y + gate_half_width, gate_z]), obs_radius),
        (np.array([gate_x, gate_y - gate_half_width, gate_z]), obs_radius),
    ]

    n_extra = np.random.randint(n_extra_range[0], n_extra_range[1] + 1)
    for _ in range(n_extra):
        for _attempt in range(50):
            radius = np.random.uniform(0.2, 0.5)
            center = np.array([
                np.random.uniform(0.3, max(goal[0] - 0.3, 0.5)),
                np.random.uniform(-2.5, 2.5),
                np.random.uniform(0.5, 2.5),
            ])
            margin = DRONE_RADIUS + radius + SAFETY_MARGIN
            if (np.linalg.norm(center - start) < margin or
                    np.linalg.norm(center - goal) < margin):
                continue
            if all(np.linalg.norm(center - c) >= r + radius + DRONE_RADIUS * 2
                   for c, r in obstacles):
                obstacles.append((center, radius))
                break

    return start, goal, obstacles


# ── Worker (one sample per call) ─────────────────────────────────────────────

HORIZON_S = 4.0  # receding-horizon window for simulate_horizon(), see worker()
CARRIER_MAX_STEPS = 400  # cap on the carrier trajectory used to source query states
GATE_SCENE_PROB = 0.35  # fraction of scenes built from random_gate_scene() instead of random_scene()
DECISION_WEIGHT_DECAY = 1.0  # meters; see _decision_weight()
DECISION_WEIGHT_FLOOR = 0.05  # baseline sampling weight for open-space states


def _decision_weight(pos, eff_obstacles, decay=DECISION_WEIGHT_DECAY, floor=DECISION_WEIGHT_FLOOR):
    """Sampling weight for a carrier-trajectory state, favoring states near an obstacle's
    effective boundary over open-space cruise."""
    if not eff_obstacles:
        return floor
    min_gap = min(float(np.linalg.norm(pos - c)) - r for c, r in eff_obstacles)
    return floor + float(np.exp(-max(0.0, min_gap) / decay))


def worker(params):
    """Generate one (scene, query state, candidate gamma) -> recedinghorizon label sample."""
    alpha_obs, n_obs_min, n_obs_max, tight_prob = params
    np.random.seed()  # re-seed each worker process

    if np.random.random() < GATE_SCENE_PROB:
        start, goal, obstacles = random_gate_scene()
    else:
        start, goal, obstacles = random_scene(
            n_obs_range=(n_obs_min, n_obs_max + 1),
            tight_prob=tight_prob,
        )

    # Use effective radii (physical + SAFETY_MARGIN) for simulation — this mirrors
    # how obstacle_cbf_node.py builds its obstacle list at inference time.
    eff_obstacles = [(c, r + SAFETY_MARGIN) for c, r in obstacles]

    carrier_gamma = np.random.uniform(*GAMMA_RANGE)
    boundary = (-1.0, 8.0, -3.0, 3.0, 0.3, 3.0)
    pos = start.copy()
    vel = np.zeros(3)
    carrier_states = [(pos.copy(), vel.copy())]
    for _ in range(CARRIER_MAX_STEPS):
        if float(np.linalg.norm(goal - pos)) < 0.3:
            break
        a_safe, _h_step, _clipped = step_dynamics(
            pos, vel, goal, eff_obstacles, carrier_gamma, boundary,
            alpha_bnd=0.5, v_max=3.0, k_p=0.5, k_v=2.0, a_max=5.0, apply_cbf=True,
        )
        vel = vel + a_safe * 0.05
        pos = pos + vel * 0.05
        carrier_states.append((pos.copy(), vel.copy()))

    # Weight by proximity to a *currently sensed* obstacle, not just a physically-present one --
    # otherwise this would bias sampling toward states the deployed node could never actually be …
    weights = np.array([_decision_weight(p, visible_obstacles(p, eff_obstacles))
                        for p, _v in carrier_states])
    weights = weights / weights.sum()
    query_idx = np.random.choice(len(carrier_states), p=weights)
    pos0, vel0 = carrier_states[query_idx]

    # Velocity resampling (2026-08-19, Phase B3 follow-up): a carrier trajectory naturally
    # decelerates AS it reacts to an obstacle, so a position sampled near an obstacle almost …
    if np.random.random() < 0.5:
        err0 = goal - pos0
        dist0 = float(np.linalg.norm(err0))
        goal_dir = err0 / dist0 if dist0 > 1e-6 else np.zeros(3)
        speed = np.random.uniform(0.0, 3.0)  # matches deployed v_max
        direction = goal_dir + np.random.normal(0.0, 0.15, size=3)
        dir_norm = float(np.linalg.norm(direction))
        direction = direction / dir_norm if dir_norm > 1e-6 else goal_dir
        vel0 = direction * speed

    # The deployed node never calls PENN at all when self._obstacles is empty (obstacle_cbf_node.py
    # short-circuits to the default gamma) -- so a query state with nothing currently sensed is a …
    obs_visible = visible_obstacles(pos0, obstacles)
    if not obs_visible:
        return None

    _risk_horizon, progress_deficit, _frac_clipped, min_h_horizon = simulate_horizon(
        pos0, vel0, goal, eff_obstacles, alpha_obs,
        boundary=boundary, horizon_s=HORIZON_S, apply_cbf=True,
    )

    module = GATModule3D(robot_radius=DRONE_RADIUS)
    robot_state = [pos0[0], pos0[1], pos0[2], vel0[0], vel0[1], vel0[2]]
    # Graph uses effective radii to match inference-time graph construction, and only the obstacles
    # currently sensed from pos0 -- matches what /arc/obstacles would actually contain at this …
    obs_list = [[c[0], c[1], c[2], r + SAFETY_MARGIN, 0.0, 0.0, 0.0] for c, r in obs_visible]
    # y[:,0] is performance (progress_deficit over the horizon, meters -- replaces low_speed_time as
    # of 2026-08-19, Phase B3: see simulate_horizon()'s docstring).
    graph = module.create_graph(
        robot=robot_state,
        obstacles=obs_list,
        goal=list(goal),
        deadlock=progress_deficit,
        risk=min_h_horizon,
    )
    graph.gamma = torch.tensor([[alpha_obs]], dtype=torch.float)

    return {
        "graph_data": {
            "x":          graph.x.cpu().numpy(),
            "edge_index": graph.edge_index.cpu().numpy(),
            "edge_attr":  graph.edge_attr.cpu().numpy(),
            "y":          graph.y.cpu().numpy(),
            "gamma":      graph.gamma.cpu().numpy(),
        }
    }


# ── Batch generation ──────────────────────────────────────────────────────────

def default_gamma_bin_edges(gamma_range=GAMMA_RANGE, split=3.5, n_low=20, n_high=20):
    """Non-uniform bin edges: `n_low` equal-width bins below `split` (same resolution as the
    pre-2026-08-23 (0.2,3.5) stratification), `n_high` bins above it."""
    g_min, g_max = gamma_range
    split = min(split, g_max)
    edges = list(np.linspace(g_min, split, n_low + 1))
    if split < g_max:
        edges += list(np.linspace(split, g_max, n_high + 1))[1:]
    return edges


def generate(
    num_samples=100000,
    num_processes=8,
    n_obs_range=(1, 8),
    gamma_range=GAMMA_RANGE,
    tight_prob=0.7,
    n_gamma_bins=20,
    gamma_bin_edges=None,
    output_prefix="data/drone_data",
):
    """Generate num_samples episodes, stratified across gamma_range."""
    if gamma_bin_edges is not None:
        edges = list(gamma_bin_edges)
    else:
        g_min, g_max = gamma_range
        bin_width = (g_max - g_min) / n_gamma_bins
        edges = [g_min + i * bin_width for i in range(n_gamma_bins + 1)]
    n_bins = len(edges) - 1

    params = []
    for i in range(num_samples):
        bin_idx = i % n_bins
        alpha = np.random.uniform(edges[bin_idx], edges[bin_idx + 1])
        params.append((alpha, n_obs_range[0], n_obs_range[1], tight_prob))

    pool = Pool(processes=num_processes)
    results = []
    n_skipped = 0
    for res in tqdm.tqdm(pool.imap_unordered(worker, params), total=len(params)):
        if res is None:
            n_skipped += 1
        else:
            results.append(res)
    pool.close()
    pool.join()

    if n_skipped:
        print(f"Skipped {n_skipped:,}/{num_samples:,} draws "
              f"({100*n_skipped/num_samples:.1f}%) with nothing sensed at the "
              f"query state -- the deployed node never calls PENN there either.")

    fname = f"{output_prefix}_{num_samples}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {len(results):,} samples → {fname}")


if __name__ == "__main__":
    # (previously imported matplotlib here purely to call .use('Agg') -- nothing in this file plots
    # anything, and the venv's include-system-site-packages fallback picks up a system matplotlib …
    np.random.seed(42)

    generate(
        # RAISED 2026-08-24 (was 100000) alongside default_gamma_bin_edges's n_high bump (8->20,
        # doubling total bins 28->40) -- keeps low-region per-bin density close to the previously- …
        num_samples=140000,
        num_processes=16,
        n_obs_range=(1, 8),
        gamma_range=GAMMA_RANGE,
        tight_prob=0.7,
        gamma_bin_edges=default_gamma_bin_edges(GAMMA_RANGE),
        output_prefix="data/drone_data_widerange2",
    )
    print("Data generation complete.")
