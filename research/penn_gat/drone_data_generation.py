"""
drone_data_generation.py

Generates PENN+GAT training data for the quadrotor CBF using a lightweight
Python drone integrator. No ROS2, no Gazebo, no safe_control dependency.

The integrator mirrors cbf_obstacle_node.py's outer loop:
  - acceleration-controlled point mass with first-order velocity tracking
    (vel += a_safe*dt, pos += vel*dt)
  - cbf_obstacle_accel: relative-degree-2 HOCBF obstacle barrier, applied
    directly to acceleration (matches a_max in the same units as the real
    actuation limit -- see cbf_obstacle_accel()'s docstring)
  - cbf_boundary: axis-aligned box barrier on the P-tracked target velocity

Output: pickle file of graph dicts compatible with nn_model/train_drone.py.

Usage:
    python drone_data_generation.py
"""

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
DEPLOY_GAMMA_RANGE = (1.3, 8.0)  # penn_gamma_min/max at inference (updated 2026-08-23,
# range-widening plan: the closed-loop controller was measured safe (0 hard
# collisions, 300 test scenes) to gamma=8.0, and the best single constant
# moved from 3.0 to 5.0 there (+11.27% outright), with the per-scene oracle
# gap roughly doubling on top (+5.04% -> +10.52%). The prior (0.5, 2.5) value
# is preserved in git history if this needs reverting.
#
# GAMMA_RANGE widened 2026-08-23 to (0.2, 9.0) to pad DEPLOY_GAMMA_RANGE's
# new interior -- same margin philosophy as the 2026-08-19 narrowing this
# reopens (from (0.1, 12.0) to (0.2, 3.5)), which was done because the old
# wide range spanned gamma 5-12 that Phase A1's offline sweep and Stage 3
# SITL both showed as unsafe/deadlock-inducing THEN, diluting the training
# distribution with irrelevant gamma and contributing to flat/near-constant
# performance-head predictions (diagnose_perf_head_vel.py). That finding
# does NOT simply invalidate this widening: this time gamma up to 8.0 is
# empirically safe on the CURRENT controller (unlike gamma 5-12 in 2026-08-19,
# which was unsafe on the controller AT THAT TIME) -- but it is the same
# class of mistake if repeated carelessly, so generate()'s sampling is
# deliberately DENSER below 3.5 and SPARSER above it (see gamma_bin_edges),
# not uniform across the full widened span, specifically to avoid diluting
# the distribution the same way. NOT narrowed to exactly (1.3, 8.0): the
# 2026-08-18 lesson about JRD blow-up at a flush boundary still applies --
# keep DEPLOY_GAMMA_RANGE strictly interior to GAMMA_RANGE on any future
# recalibration of either.
GAMMA_RANGE   = (0.2, 9.0)
UNSAFE_PROB   = 0.20   # fraction of episodes run WITHOUT CBF so risk > 0 labels appear

# ── Sensing model — mirrors lidar_obstacle_publisher_node.py's defaults ──────
# (2026-08-20, recursive-feasibility redesign, Phase 0): every simulated
# actor -- training-data generation, the fixed-gamma baseline, and the
# adaptive policy alike -- was previously omniscient: obstacle avoidance and
# the network's graph input both used the full scene obstacle list
# regardless of distance. Real hardware is not: the Livox-based perception
# pipeline only ever publishes obstacles within max_range_m of the robot AND
# inside a world-frame z-band (z_min_m/z_max_m -- excludes floor/ceiling
# clutter, not a sensor-relative angular FOV; the sensor's own 360-deg
# horizontal FOV means no angular filtering is needed). Values below match
# that node's declared defaults exactly so sim and hardware share one
# sensing envelope instead of two.
#
# min_range_m (0.2 on the real node) is deliberately NOT modeled here: it
# exists to reject near-sensor return noise on individual lidar points, not
# to make a tracked obstacle vanish -- by the time an obstacle's centroid
# would be inside 0.2m the vehicle is already deep inside SAFETY_MARGIN
# (0.5m), so it's irrelevant to this point-mass approximation.
SENSOR_MAX_RANGE = 8.0
SENSOR_Z_MIN = 0.1
SENSOR_Z_MAX = 2.5


def visible_obstacles(pos, obstacles):
    """Filter `obstacles` (list of (center, radius)) down to what a
    perception pipeline with this sensing envelope could actually report
    from `pos` right now (distance measured to the obstacle's surface, so
    the filter behaves consistently whether `radius` is a raw physical
    radius or an already-margin-inflated effective radius).

    Used everywhere a controller or the network makes a decision -- the CBF
    avoidance correction and any graph/network input. Ground-truth collision
    scoring (h_step in step_dynamics) is NEVER filtered this way: an
    unsensed obstacle can still be hit, that's the entire point of modeling
    this at all.
    """
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
    """Relative-degree-2 HOCBF obstacle avoidance, applied directly to
    acceleration (the vehicle's real control input) instead of to the
    P-tracked target velocity.

    h = ||p-c||^2 - r^2 is relative-degree 2 in a point-mass-with-
    acceleration-input plant (two derivatives of h reach a). Enforces the
    critically-damped-style HOCBF condition
        hddot + 2*gamma*hdot + gamma^2*h >= 0
    reusing the single scalar gamma for both poles so this stays a drop-in
    replacement for the old rel-degree-1 velocity-space gain PENN adapts.

    Why this replaces the old velocity-space cbf_obstacle() call: that
    function corrected v_safe *before* the k_v acceleration-tracking law,
    so its correction was realized through k_v*(v_safe-v) and then clamped
    to a_max only as an afterthought -- a_max could never meaningfully
    interact with gamma because the correction it was clamping had already
    been computed as if velocity were the true control input. Applying the
    HOCBF correction here, after the tracking law, means a high gamma's
    required correction competes with a_max in the same units, in the same
    step -- the actuation-limit trade-off Input-Constrained CBF methods
    (Kim et al. 2025) depend on, and which was previously structurally
    absent (see barrier-label-collapse plan, Stage 1).

    CONVENTION NOTE (2026-08-19): the correction this function returns
    *grows* with gamma (verified directly: correction norm ~0.65 m/s^2 at
    gamma=0.5 vs ~8.6 m/s^2 at gamma=10 on a fixed near-obstacle test
    state) -- the opposite of the old rel-degree-1 cbf_obstacle()'s
    convention (low gamma = reacts early/conservative, high gamma = reacts
    late/aggressive). Here low gamma is the weak/under-reacting end and
    high gamma the strong end, which can itself become unsafe once it
    saturates a_max and the clipped correction causes overshoot. Treat any
    old "prefer low gamma when uncertain" reasoning as inapplicable to this
    function specifically.

    Returns (a_safe, feasible) -- feasible False only on deep penetration
    (h < -radius^2), mirroring the old function's semantics.
    """
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
    """One 'dt-normalized' control step: goal+boundary tracking in velocity
    space, obstacle avoidance in acceleration space, a_max clip. Shared by
    simulate_episode() (whole-episode-from-start) and simulate_horizon()
    (receding-horizon from an arbitrary state) so the two can't drift apart
    the way separately-maintained copies of this project's CBF math already
    have once before (see obstacle_cbf_node.py vs cbf_obstacle_node.py
    history). Returns (a_safe, h_step, clipped) -- does NOT integrate
    pos/vel; callers do that with their own dt.
    """
    err = goal - pos
    v_nom = k_p * err
    speed = float(np.linalg.norm(v_nom))
    if speed > v_max:
        v_nom = v_nom / speed * v_max

    v_safe = cbf_boundary(pos, v_nom, boundary, alpha_bnd)

    # Ground truth: did we actually get close to something, full obstacle
    # set, regardless of what's currently sensed. A physically real obstacle
    # causes a physically real violation whether or not it was perceived.
    h_step = np.inf
    for center, radius in obstacles:
        n = pos - center
        h = float(np.dot(n, n)) - radius ** 2
        if h < h_step:
            h_step = h

    a_safe = k_v * (v_safe - vel)

    if apply_cbf:
        # Avoidance can only react to what's currently sensed -- matches
        # obstacle_cbf_node.py, which only ever has whatever the (filtered)
        # /arc/obstacles topic last published in self._obstacles.
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
    """
    Run one drone episode with the given CBF parameters.

    v_max/k_p/k_v/a_max default to the exact values declared in
    cbf_obstacle_node.py's ROS parameters, and the position update below
    mirrors its _tick()/outer-loop structure: the CBF layer only ever
    proposes a *target* safe velocity (v_safe); the vehicle's actual
    velocity chases it through a first-order acceleration-tracking law
    (a_safe = k_v*(v_safe - v_actual), clamped to a_max) rather than
    snapping to it instantly.

    Earlier version of this function did `pos = pos + v_safe * dt` directly
    -- a zero-inertia point-mass assumption where every CBF correction is
    realized in the very same timestep it's requested. That let training
    rollouts "see" large alpha_obs as always safe (a point-mass can swerve
    infinitely fast, so the closed-form velocity CBF is safe by
    construction under its own dynamics), while the real quadrotor -- which
    has to accelerate its way to v_safe against real momentum and thrust
    limits -- cannot realize a sharp last-second correction in time. This
    mismatch was diagnosed 2026-08-18 after PENN-selected alpha_obs=10
    produced two real min_obs_h<0 violations in PX4 SITL, both with zero
    epistemic-gate rejection (i.e. not a coverage/OOD gap the ensemble could
    ever flag -- every member was trained on the same unrealistic dynamics
    and agreed confidently). Adding the acceleration-tracking step here
    makes the label-generating rollouts dynamically consistent with the
    actual deployed control loop.

    Obstacle avoidance itself was moved from velocity space to acceleration
    space 2026-08-18 (see barrier-label-collapse plan, Stage 1): the old
    cbf_obstacle() corrected v_safe *before* the k_v tracking law below, so
    a_max only ever clamped an already-computed correction rather than
    genuinely competing with gamma inside the same constraint. See
    cbf_obstacle_accel()'s docstring for the full rationale. a_max is left
    at its existing default (5.0) deliberately -- it mirrors the real
    vehicle's CBF_accel_bound_ z-axis sanity check in
    single_vehicle_cbf_client.hpp, and raising it here without a matching
    C++ change would silently desync training from what the real client
    actually accepts.

    Args:
        apply_cbf: if False, CBF corrections are skipped so the drone may
                   penetrate obstacles (gives risk > 0 training labels).

    Returns:
        min_h        (float) — minimum barrier value over all obstacles & steps.
                     Diagnosed 2026-08-18 as a bad *label* despite being a
                     correct thing to log: under an active CBF, h obeys
                     hdot=-gamma*h whenever the constraint binds, so
                     h(t)=h0*exp(-gamma*t) -> 0+ for ANY gamma. gamma sets
                     the approach *rate*, not the floor min_h settles at, so
                     this value alone carries almost no gamma signal (see
                     barrier-label-collapse plan). Kept for backward
                     comparison/logging only -- do not train on it.
        viol_integral(float) — integral of max(0,-h) dt over all obstacles &
                     steps. Unlike min_h this keeps accumulating for as long
                     as the drone stays inside the margin, so it separates a
                     brief graze from a sustained violation and -- per the
                     same-day label-candidate sweep -- is actually
                     discriminative across gamma. This is the new risk label.
        t_goal       (float) — seconds to reach the goal, capped at
                     max_steps*dt if never reached. The performance label
                     (replaces the old deadlock_time head, which never
                     separated gamma either: actuator saturation never once
                     fired across the validation sweep).
        reached_goal (bool)
        frac_accel_clipped (float) — fraction of steps where the obstacle-
                     avoidance acceleration correction exceeded a_max and
                     was clipped. Instrumentation for the Stage-1 gate: if
                     this stays ~0 across the whole gamma range, a_max has
                     no leverage at the current v_max and gamma still can't
                     be two-sided. Not a training label.
    """
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
    """Simulate forward from an arbitrary (pos0, vel0) state for horizon_s
    seconds under a fixed candidate gamma, and return receding-horizon
    performance/risk metrics.

    This is the Stage-2 replacement for simulate_episode()'s whole-episode-
    from-start labels (barrier-label-collapse plan): the old labels
    answered "if the drone starts at rest near the origin and flies this
    whole scene with gamma, what happens by the end?" but the deployed node
    queries PENN mid-course, from whatever (pos, vel) it currently has --
    at least one train/query state-distribution mismatch this project had
    not yet addressed. Kim et al. (arXiv:2409.14616) label from the current
    state over a forward window, not from a fixed start; this mirrors that
    choice.

    Returns:
        risk_horizon (float) — integral of max(0,-h) dt over the horizon,
                     same accumulation rule as simulate_episode's
                     viol_integral, just windowed to [0, horizon_s] instead
                     of the whole episode.
        progress_deficit (float) — meters of goal-directed progress lost
                     over the horizon relative to the theoretical best case
                     (flying straight at v_max the whole time, capped by the
                     actual distance to goal). Low is good (making
                     consistent progress); high means stalled/deadlocked/
                     routing away from the goal -- can happen at either end
                     of gamma: too low under-reacts and drifts into a slow
                     detour, too high can command a correction that
                     saturates a_max and oscillates (see
                     cbf_obstacle_accel()'s convention note -- gamma's
                     effect here is not simply monotonic).

                     NOT residualized against an obstacle-free reference,
                     despite that looking like the obvious fix for this
                     label's high between-state variance (2026-08-20,
                     Stage 2, barrier-label-collapse plan's scene-level
                     reframe): tried and measured NOT to work. Subtracting
                     the obstacle-free/no-CBF deficit for the same
                     (pos0, vel0, goal) left the between/within-gamma split
                     essentially unchanged (83.8%->81.8% between-state) and
                     introduced a new 57-64% exact-zero mass -- the same
                     MSE-trap failure mode the old low_speed_time label had,
                     reintroduced by a different mechanism (most sampled
                     states genuinely aren't near an obstacle, so the
                     residual floors to ~0 for most of the dataset). The
                     remaining between-state variance is NOT primarily
                     P-controller geometry; it tracks something more like
                     per-encounter severity, which subtracting a straight-
                     line reference doesn't remove. Left un-fixed here;
                     Stage 0's held-out ridge-regression check
                     (eval_scene_distribution.py-adjacent diagnostics) shows
                     the underlying oracle-gamma signal IS learnable from
                     scene features despite this label's variance profile,
                     so the fix -- if one is still needed after retraining
                     with the other two Stage-2 changes -- likely belongs at
                     the loss/architecture level (e.g. a per-state relative
                     ranking loss across gamma) rather than another label
                     transform.

                     REPLACES the old low_speed_time label (2026-08-19,
                     Phase B3, barrier-label-collapse plan): low_speed_time
                     only accumulated while goal-directed speed was below a
                     hard SPEED_THRESHOLD=0.15 m/s cutoff, so any state that
                     never dipped below that cutoff registered as an exact
                     zero regardless of how much gamma actually cost in
                     travel efficiency -- diagnosed as 91.3% exact-zero over
                     a 4000-sample replay of worker()'s own sampling
                     (diagnose_label_dist.py), which let the performance
                     head minimize its loss by predicting a near-constant
                     and never actually fitting gamma's effect
                     (diagnose_perf_head.py / diagnose_perf_head_vel.py:
                     predicted-vs-true correlation was NEGATIVE, and the
                     predicted curve collapsed to a ~0.001 spread at real
                     flight velocity). progress_deficit is a single
                     continuous scalar per rollout with no hard threshold,
                     so it carries a nonzero, informative value for
                     essentially any rollout that isn't a perfect
                     straight-line max-speed transit -- not just the rare
                     ones that crossed the old cutoff.
        frac_clipped (float) — fraction of horizon steps where a_max bound.
        min_h_horizon (float) — minimum instantaneous barrier value h seen
                     anywhere in the horizon (0.0 if the horizon has zero
                     steps, e.g. already at the goal). Added 2026-08-20
                     after diag_horizon_blindspot.py found risk_horizon's
                     accumulated-violation-TIME integral and "does h ever go
                     negative" are not the same question: a brief, shallow
                     graze contributes almost nothing to the integral (so it
                     passes a tight cvar_boundary gate easily) while still
                     being exactly the min_h<0 event the deployed system
                     calls a safety violation. On 18 held-out scenes that
                     went unsafe post-retrain, ground truth risk_horizon at
                     the pre-violation query state/gamma was itself below
                     cvar_boundary on 67% of them -- not a model calibration
                     error, a metric that structurally can't bound this.
                     min_h_horizon exists so a selection rule can gate on
                     "never crosses zero" directly instead of only on the
                     integral.
    """
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
    """
    Sample a random 3D scene: start, goal, and N sphere obstacles.

    With probability near_goal_prob the robot is placed near the goal
    (fills the OOD gap seen when the drone holds position at a waypoint).

    Returns:
        start    : np.ndarray [3]
        goal     : np.ndarray [3]
        obstacles: list of (center [3], radius float)  — physical radii, no margin
    """
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
            # For near-goal scenes the path is short (0.5–2 m) so path-based
            # placement rarely clears the start/goal margin.  Instead scatter
            # obstacles at a fixed clearance distance from start in a random
            # direction — this always produces at least one close obstacle.
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
    """Sample a scene with a deliberate two-obstacle "gate" directly on the
    path, plus optional extra scattered obstacles elsewhere.

    Added 2026-08-19 (Stage 2 follow-up, barrier-label-collapse plan):
    random_scene() places every obstacle independently via rejection
    sampling, so a genuinely narrow gap between a *pair* of obstacles is
    incidental, not a targeted archetype -- an offline check of the
    retrained Stage-2 checkpoint on a hand-built two-obstacle gate showed
    the model extrapolating there (epistemic gate correctly flagged
    low-gamma as OOD at JRD=0.064 vs a 0.035 threshold) and disagreeing
    with the ground-truth point-mass sweep on which gamma is actually safe.
    That specific geometry -- two obstacles forming a corridor gate -- is
    also exactly what the real deployed course (BASE_OBSTACLES in
    cbf_sitl_test_launch.py) is built from, so it's the single most
    important archetype to have in-distribution, not an edge case.

    gate_half_width spans from clearly-open to clearly-impossible
    (obstacle surfaces overlapping) so both regimes Stage 1's gate sweep
    found -- flat/no-tradeoff on a passable gate, genuine two-sided cost on
    an impossible one -- are represented with real density, not just
    incidentally sampled.

    Returns the same (start, goal, obstacles) shape as random_scene().
    """
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
    """Sampling weight for a carrier-trajectory state, favoring states near
    an obstacle's effective boundary over open-space cruise.

    Added 2026-08-19 (Phase B3, barrier-label-collapse plan): the old
    uniform-over-the-whole-trajectory query sampling meant most training
    samples landed in open space, where gamma genuinely doesn't matter (any
    candidate produces an identical safe, fast rollout) -- diluting the
    minority of samples where gamma actually changes the outcome. A 4000-
    sample replay of the old sampling (diagnose_label_dist.py) found 91.3%
    exact-zero performance labels as a direct consequence. This weight is an
    exponential decay in min-gap-to-nearest-obstacle-boundary (0 at/inside
    the boundary, decaying over `decay` meters), plus a small floor so
    open-space states are still sampled sometimes -- the deployed node does
    sometimes query PENN in open space, and the model shouldn't misbehave
    there either.
    """
    if not eff_obstacles:
        return floor
    min_gap = min(float(np.linalg.norm(pos - c)) - r for c, r in eff_obstacles)
    return floor + float(np.exp(-max(0.0, min_gap) / decay))


def worker(params):
    """Generate one (scene, query state, candidate gamma) -> receding-
    horizon label sample.

    Stage 2 of the barrier-label-collapse plan: the old worker() ran one
    whole episode from the scene's start under the sample's gamma and
    labeled the *entire* episode -- but the deployed node queries PENN
    mid-course, from whatever (pos, vel) it currently has. That's a
    train/query state-distribution mismatch: the model was only ever asked
    "what happens over a full flight starting at rest near the origin?",
    never "what happens over the next few seconds from here?" This worker
    instead:
      1. Flies the scene once under an independently-sampled *carrier*
         gamma to reach a diverse, dynamically-reachable set of states
         (not just the start).
      2. Picks a point along that carrier trajectory as the query state
         (pos0, vel0), weighted toward states near an obstacle's effective
         boundary rather than uniform over the whole trajectory (Phase B3,
         2026-08-19 -- see _decision_weight()'s docstring for why uniform
         sampling produced degenerate, mostly-zero performance labels).
      3. Labels that state under the sample's actual *query* gamma via
         simulate_horizon() -- a short forward rollout, matching what the
         node itself does when it evaluates a gamma candidate from the
         robot's current state.
    Carrier and query gamma are independent draws so the state distribution
    isn't correlated with the label gamma.
    """
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

    # Weight by proximity to a *currently sensed* obstacle, not just a
    # physically-present one -- otherwise this would bias sampling toward
    # states the deployed node could never actually be in when it queries
    # PENN (obstacle physically nearby but outside the sensing envelope).
    weights = np.array([_decision_weight(p, visible_obstacles(p, eff_obstacles))
                        for p, _v in carrier_states])
    weights = weights / weights.sum()
    query_idx = np.random.choice(len(carrier_states), p=weights)
    pos0, vel0 = carrier_states[query_idx]

    # Velocity resampling (2026-08-19, Phase B3 follow-up): a carrier
    # trajectory naturally decelerates AS it reacts to an obstacle, so a
    # position sampled near an obstacle almost always comes with an
    # already-slow velocity -- a diagnose_velocity_coverage.py check found
    # only 3.1% of gap<0.5m carrier states have speed>1.5 m/s and 0.2% have
    # speed>2.0 m/s. The deployed node queries PENN periodically
    # (penn_update_ticks, ~1s) and can catch the drone still at cruise speed
    # just as it's approaching an obstacle, before any correction has taken
    # effect -- an (position, velocity) combination the carrier-only
    # sampling above almost never produces. With 50% probability, replace
    # the carrier's actual velocity with an independently-drawn one at the
    # same position: magnitude uniform in [0, v_max], direction toward the
    # goal with heading noise (matching what the k_p tracking law actually
    # produces, not an arbitrary 3D vector).
    if np.random.random() < 0.5:
        err0 = goal - pos0
        dist0 = float(np.linalg.norm(err0))
        goal_dir = err0 / dist0 if dist0 > 1e-6 else np.zeros(3)
        speed = np.random.uniform(0.0, 3.0)  # matches deployed v_max
        direction = goal_dir + np.random.normal(0.0, 0.15, size=3)
        dir_norm = float(np.linalg.norm(direction))
        direction = direction / dir_norm if dir_norm > 1e-6 else goal_dir
        vel0 = direction * speed

    # The deployed node never calls PENN at all when self._obstacles is
    # empty (obstacle_cbf_node.py short-circuits to the default gamma) --
    # so a query state with nothing currently sensed is a state the network
    # is never actually run from. Skip it rather than emit a degenerate
    # zero-obstacle graph sample the deployed system would never produce.
    obs_visible = visible_obstacles(pos0, obstacles)
    if not obs_visible:
        return None

    _risk_horizon, progress_deficit, _frac_clipped, min_h_horizon = simulate_horizon(
        pos0, vel0, goal, eff_obstacles, alpha_obs,
        boundary=boundary, horizon_s=HORIZON_S, apply_cbf=True,
    )

    module = GATModule3D(robot_radius=DRONE_RADIUS)
    robot_state = [pos0[0], pos0[1], pos0[2], vel0[0], vel0[1], vel0[2]]
    # Graph uses effective radii to match inference-time graph construction,
    # and only the obstacles currently sensed from pos0 -- matches what
    # /arc/obstacles would actually contain at this instant.
    obs_list = [[c[0], c[1], c[2], r + SAFETY_MARGIN, 0.0, 0.0, 0.0] for c, r in obs_visible]
    # y[:,0] is performance (progress_deficit over the horizon, meters --
    # replaces low_speed_time as of 2026-08-19, Phase B3: see
    # simulate_horizon()'s docstring).
    #
    # y[:,1] (risk) SWAPPED 2026-08-20 (recursive-feasibility redesign,
    # Phase 2 follow-up) from risk_horizon (the accumulated-violation-time
    # integral) to min_h_horizon directly. Motivated by a live experiment:
    # gating on ground-truth min_h_horizon (oracle_min_h mode) and halving
    # the deployed query cadence together cut the margin-violation rate in
    # half (12%->6%), while cadence alone made zero difference (even made
    # things slightly worse) to the SHIPPED risk_horizon-gated policy --
    # because checking a metric that can't represent the failure mode more
    # often doesn't help (see the accumulated-violation-time-gate pattern
    # note). min_h_horizon is the quantity that actually distinguishes "the
    # constraint is ever crossed" from "crossed briefly, integrates away to
    # nothing" -- the same fix diag_horizon_blindspot.py already applied at
    # the oracle level, now pushed into the label the network is trained on
    # directly. Sign convention is now INVERTED from the old risk label:
    # LOW/NEGATIVE = dangerous, HIGH/POSITIVE = safe, not floored at zero
    # (see AdaptivePolicy.select()'s and cccp_calibrate_drone.py's matching
    # updates -- both read this head's prediction and must agree on the
    # sign).
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
    """Non-uniform bin edges: `n_low` equal-width bins below `split` (same
    resolution as the pre-2026-08-23 (0.2,3.5) stratification), `n_high`
    bins above it. Added for the range-widening plan so the newly added
    gamma>3.5 mass is sparser per unit gamma than the low range, rather
    than uniform over the whole (0.2, 9.0) span -- a naive uniform
    stratification would put more than half the training samples above 3.5,
    repeating the 2026-08-19 dilution failure mode this project already hit
    once with a wide range (0.1, 12.0).

    n_high RAISED 2026-08-24 (was 8, giving ~4x sparser density than the
    low region -- 0.6875 gamma/bin vs 0.165). penn_prediction_quality_check
    (see single_vehicle_cbf_rate_arc/client_lib/tools/) found the risk
    head's predictions turn BACKWARDS above ~gamma=4.5 (predicting higher
    gamma as more dangerous, contradicting the ground-truth safety sweep's
    0 hard collisions from 3.2-10.0) in 51% of scenes, and the resulting
    cvar-gate false-rejection throttles the deployed selector's effective
    range in 58% of scenes -- diagnosed as sparse-region extrapolation, not
    a real learned effect. 20 roughly matches the low region's per-bin
    density (paired with the num_samples bump in generate()'s __main__
    call, below) rather than fixing the split point itself -- DEPLOY_GAMMA_
    RANGE=(1.3,8.0) already sits mostly above the old split=3.5, so the
    region that needed density was specifically the one that got starved."""
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
    """Generate num_samples episodes, stratified across gamma_range.

    gamma_bin_edges, if given, overrides n_gamma_bins with an explicit,
    non-uniform bin-edge list (see default_gamma_bin_edges) -- added
    2026-08-23 for the range-widening plan. Omitting it reproduces the
    original uniform-bin behavior exactly."""
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
    # (previously imported matplotlib here purely to call .use('Agg') --
    # nothing in this file plots anything, and the venv's
    # include-system-site-packages fallback picks up a system matplotlib
    # built against numpy 1.x, which crashes against the venv's numpy 2.x.
    # Dropped as dead code rather than chasing the numpy/matplotlib pin.)
    np.random.seed(42)

    generate(
        # RAISED 2026-08-24 (was 100000) alongside default_gamma_bin_edges's
        # n_high bump (8->20, doubling total bins 28->40) -- keeps low-region
        # per-bin density close to the previously-validated ~3571/bin
        # (100000/40=2500 alone would have diluted it further; 140000/40=3500
        # holds it roughly steady while giving the high region real density
        # for the first time).
        num_samples=140000,
        num_processes=16,
        n_obs_range=(1, 8),
        gamma_range=GAMMA_RANGE,
        tight_prob=0.7,
        gamma_bin_edges=default_gamma_bin_edges(GAMMA_RANGE),
        output_prefix="data/drone_data_widerange2",
    )
    print("Data generation complete.")
