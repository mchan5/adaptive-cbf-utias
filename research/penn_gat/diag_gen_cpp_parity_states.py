"""Generates a batch of random (pos, vel, goal, obstacles) states, computes the Python
reference's selected gamma for each via AdaptivePolicy.select() (the exact validated logic), …"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nn_model"))

import eval_scene_distribution as E  # noqa: E402
import drone_data_generation as G  # noqa: E402

N_STATES = 300
MAX_OBS = 16
# The flat exported-tensor weights the C++ node actually loads.
DEPLOY_PT = os.path.join(os.path.dirname(__file__), "..", "single_vehicle_cbf_rate_arc",
                         "nn_model", "checkpoint", "gat_penn_weights.pt")


def fnv1a64(path):
    """64-bit FNV-1a over a file's bytes. Not cryptographic -- a cheap cross-language content
    stamp (mirrored in penn_gamma_parity_check.cpp) for staleness detection only."""
    h = 0xCBF29CE484222325
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            for byte in chunk:
                h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "single_vehicle_cbf_rate_arc",
                        "nn_model", "checkpoint", "cpp_parity_states.pt")


def sample_query_state(rng, scene):
    """Sample a plausible (pos, vel) query state along a scene's flight envelope, biased toward
    near-obstacle states the same way training does -- covers both the far-field (epistemic- …"""
    start, goal, obstacles = scene
    t = rng.uniform(0.0, 1.0)
    pos = start + t * (goal - start) + rng.normal(0, 0.4, size=3)
    pos[2] = np.clip(pos[2], 0.3, 3.0)
    speed = rng.uniform(0.0, E.V_MAX)
    direction = (goal - pos)
    n = np.linalg.norm(direction)
    direction = direction / n if n > 1e-6 else np.zeros(3)
    direction = direction + rng.normal(0, 0.2, size=3)
    n2 = np.linalg.norm(direction)
    direction = direction / n2 if n2 > 1e-6 else np.zeros(3)
    vel = direction * speed
    return pos, vel, goal, obstacles


def main():
    rng = np.random.default_rng(20260822)
    # lcb_k=0.25: the deployed calibrated value (recalibrated 2026-08-22 on the closed-loop campaign
    # after point-mass-only lcb_k=1.50 measured worse there -- see reproduce_headline.py's …
    LCB_K = float(os.environ.get("LCB_K", "0.0"))
    # FALLBACK_STEP: config yaml -> penn_fallback_step (deployed 0.5).
    FALLBACK_STEP = float(os.environ.get("FALLBACK_STEP", "0.5"))
    policy = E.AdaptivePolicy(E.DEFAULT_CKPT, E.DEFAULT_THRESH, gate_mode="network",
                              lcb_k=LCB_K, fallback_step=FALLBACK_STEP,
                              fallback_mode="ratchet")

    scenes = E.make_holdout_scenes(60, seed=20261001)  # disjoint from every other seed used this session

    pos_l, vel_l, goal_l, gamma_l, prev_gamma_l = [], [], [], [], []
    obs_centers = np.full((N_STATES, MAX_OBS, 3), 0.0, dtype=np.float32)
    obs_radii = np.full((N_STATES, MAX_OBS), -1.0, dtype=np.float32)

    n_written = 0
    attempts = 0
    while n_written < N_STATES and attempts < N_STATES * 5:
        attempts += 1
        scene = scenes[rng.integers(0, len(scenes))]
        pos, vel, goal, obstacles = sample_query_state(rng, scene)
        visible = G.visible_obstacles(pos, obstacles)
        if not visible or len(visible) > MAX_OBS:
            continue
        # skip empty-obstacle states -- select() short-circuits before ever touching the model there
        # Each parity state is standalone -- reset the ratchet and hand select() an explicit …
        prev_gamma = float(rng.uniform(E.DEPLOY_GAMMA_MIN, E.DEPLOY_GAMMA_MAX))
        policy.reset()
        gamma = policy.select(pos, vel, goal, obstacles, prev_gamma=prev_gamma)

        # BUG (found 2026-08-22 while chasing a parity mismatch): this used to save `obstacles` (the
        # full, unfiltered scene list) here, while `visible` (already computed above for the …
        pos_l.append(pos.astype(np.float32))
        vel_l.append(vel.astype(np.float32))
        goal_l.append(goal.astype(np.float32))
        gamma_l.append(float(gamma))
        prev_gamma_l.append(prev_gamma)
        for k, (c, r) in enumerate(visible):
            obs_centers[n_written, k] = c.astype(np.float32)
            obs_radii[n_written, k] = float(r)
        n_written += 1

    print(f"Generated {n_written} states ({attempts} attempts) "
          f"(fallback fired on {policy.n_fallback}/{policy.n_queries} queries, "
          f"{100*policy.n_fallback/max(policy.n_queries,1):.1f}%)")

    payload = {
        "n_states": n_written,
        "max_obs": MAX_OBS,
        "pos": torch.from_numpy(np.stack(pos_l)),
        "vel": torch.from_numpy(np.stack(vel_l)),
        "goal": torch.from_numpy(np.stack(goal_l)),
        "prev_gamma": torch.tensor(prev_gamma_l, dtype=torch.float32),
        "obs_centers": torch.from_numpy(obs_centers[:n_written]),
        "obs_radii": torch.from_numpy(obs_radii[:n_written]),
        "expected_gamma": torch.tensor(gamma_l, dtype=torch.float32),
        "epistemic_threshold": policy.ep_thr,
        "cvar_boundary": policy.risk_thr,
        "lcb_k": LCB_K,
        "fallback_step": FALLBACK_STEP,
        "gamma_min": E.DEPLOY_GAMMA_MIN,
        "gamma_max_candidate": E.DEPLOY_GAMMA_MAX,
        "gamma_steps": E.DEPLOY_STEPS,
        # Staleness stamp: hash of the .pt the C++ node loads, at generation time.
        "weights_fnv1a64": f"{fnv1a64(DEPLOY_PT):#018x}",  # hex string: torch pickle can't hold uint64
        "checkpoint_name": os.environ.get("CHECKPOINT_NAME", "Quadrotor3D_gat"),
    }
    print(f"weights stamp (fnv1a64 of {os.path.relpath(DEPLOY_PT)}): "
          f"{payload['weights_fnv1a64']}  ckpt={payload['checkpoint_name']}  "
          f"gamma=[{E.DEPLOY_GAMMA_MIN},{E.DEPLOY_GAMMA_MAX}]x{E.DEPLOY_STEPS}  lcb_k={LCB_K}")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(payload, OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
