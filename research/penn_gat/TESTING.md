# Testing / Verification & Validation Procedure

This document formalizes the testing pyramid this project actually uses. Most of it already
existed as fragmented scripts before this was written (2026-08-24) — this consolidates them
into one named, ordered procedure with explicit pass/fail gates, and adds the two tiers that
were genuinely missing (Tier 0 unit tests, Tier 4b mechanism stress test).

**Why this exists**: during the 2026-08-24 authority-tuning session, two controller configs
passed every offline check cleanly (0 hard collisions across 950 episodes spanning three
scene distributions) and then failed real SITL — one with an actual collision, one with a
control-loop divergence that an 8-scene spot-check only caught because the logs were read by
hand, not because any aggregate metric flagged it. That is exactly the failure mode a
documented procedure with stated sample-size requirements exists to prevent: an offline pass
being mistaken for a validated result.

## The pyramid

| Tier | What | Fidelity | Speed | Script/binary |
|---|---|---|---|---|
| 0 | Component unit tests | N/A (deterministic logic) | seconds | `client_lib/test/*_test.cpp` (gtest) |
| 1 | Cross-implementation parity | N/A (consistency, not physics) | seconds | `penn_gamma_parity_check.cpp` |
| 2 | Point-mass statistical | Low | fast, large-N | `reproduce_headline.py`, `eval_scene_distribution.py` |
| 3 | Body-rate closed-loop | Medium | fast (sim clock), medium-N | `body_rate_closed_loop_diag.cpp` |
| 4 | SITL (Gazebo + PX4) | High | real-time, small/medium-N | `cbf_rate_arc_sitl_test_launch.py` + `tools/gen_sitl_obstacle_scenes.py` + `tools/parse_sitl_results.py` |
| 4b | Targeted mechanism stress test | High (same as Tier 4) | real-time but short episodes | `tools/gen_sitl_obstacle_scenes.py --empty` + `tools/run_arming_transient_sweep.sh` + `tools/parse_arming_transient.py` |

Each tier's pass is **necessary, not sufficient** for the tier above it. In particular:

> **Tier 3's known limitation.** `body_rate_closed_loop_diag.cpp`'s rigid-body integrator
> assumes commanded thrust/body-rate is achieved **instantaneously** (stated in that file's
> own header comment, matching the barrier derivation's own idealization). This is
> indistinguishable from reality at low-to-moderate controller authority, but the 2026-08-24
> session found it breaks down at higher authority: two configs (`nom_accel_max`=13, 14) passed
> Tier 3 with 0/950 hard collisions and then failed Tier 4. **A Tier 3 pass must never be
> reported as a validated result on its own for any config whose authority hasn't already
> cleared Tier 4** — it is a necessary screening step, not evidence of real-world safety.

### Tier 0 — Component unit tests

Deterministic, no ROS init, no QP solve required where avoidable. Gate: **100% pass**, run on
every change that touches `client_lib/`. Location: `client_lib/test/`, built via
`ament_add_gtest` under `if(BUILD_TESTING)` in `CMakeLists.txt`, run with
`colcon test --packages-select single_vehicle_cbf_rate_arc`.

Current coverage (deliberately modest — a handful of high-value tests, not exhaustive):
- `core_sanity_check_test.cpp` — the physical-sanity scenarios `core_sanity_check.cpp`
  already checked by eye, now as real `EXPECT_*` assertions.
- `penn_gamma_selector_gate_test.cpp` — `PennGammaSelector`'s proximity-gate logic in
  isolation (synthetic obstacle input, no QP), including the 2026-08-24 regression case
  (gate must hold gamma fixed inside `penn_proximity_gate_m`).
- `nominal_pd_clamp_test.cpp` — `nom_accel_max` norm-clamp behavior on the nominal PD law.

### Tier 1 — Cross-implementation parity

Gate: full batch agreement (currently 300/300) between Python (`eval_scene_distribution.py`'s
`AdaptivePolicy`) and C++ (`PennGammaSelector`) gamma selection, byte-exact. Only meaningful
while PENN is enabled; skip while `penn_enabled: false` (current deploy state).

### Tier 2 — Point-mass statistical validation

Gate: **0 hard collisions** on the disjoint `TEST_SEEDS` set (n≥350), using the established
`CALIBRATION_SEEDS`/`TEST_SEEDS` split (`reproduce_headline.py:53-54`). Calibration seeds may
be used for tuning; TEST_SEEDS must be run once, unmodified, no re-tuning against it.

### Tier 3 — Body-rate closed-loop validation

Same seeds/gate convention as Tier 2, run through `body_rate_closed_loop_diag.cpp`'s full
12-state QP + rigid-body integration instead of the point-mass simulator. See the boxed
limitation above — this tier screens out bad configs cheaply, it does not clear a config for
deployment on its own.

### Tier 4 — SITL validation

Gate defined via the **rule of three**: for `n` independent trials with 0 observed failures,
the upper ~95% confidence bound on the true failure rate is ≈3/n (standard result for
rare-event/zero-failure testing).

| n | upper 95% bound on true failure rate |
|---|---|
| 8 | ~37.5% |
| 30 | ~10% |
| 50 | ~6% |
| 100 | ~3% |

**n≥50 (≈6% bound) is the minimum sample size for a config to be called "SITL-validated" in
this project going forward.** n=8 (this project's convention through 2026-08-24) is
relabeled **"spot-check"** — informative for catching gross failures cheaply and deciding
whether to invest in a full campaign, but not a validation claim on its own.

> **This is not a hypothetical.** The same day this rule was written, an authority-tuning
> config `(12,6,6,gamma=7.0)` passed an 8-scene spot-check (0/8 hard collisions) and was
> deployed. A subsequent n=50 battery — motivated by nothing more than applying this
> document's own stated minimum — found **2/50 hard collisions (4%, 95% CI [0.5%,
> 13.7%])**, a different, previously-undetected failure mode from the one the spot-check
> process had been reasoning about all day. The original `(8,4,4,gamma=8)` baseline was then
> run through the same n=50 battery and held: **0/50, 95% upper bound 7.1%** — the first
> config in this project to actually clear this gate. Root-causing the 2 collisions found two
> distinct, confirmed bugs in `rate_autopilot_core.cpp` (a certification-routing bug and a
> memoryless infeasible-recovery fallback with no cooldown) -- fixing both and re-running the
> same n=50 battery brought it to 1/50, with one mechanism fully resolved and the other
> reduced 97% in severity but not fully closed (best guess: a discrete-time CBF sampling-rate
> limit, not confirmed). See `config/params_single_vehicle_cbf_rate_arc.yaml`'s
> `nom_accel_max` history and `results/frozen_authority_20260824/MANIFEST.md` for the full
> incident, including a separate contamination bug (an orphaned `synthetic_obstacle_publisher`
> from an earlier SIGKILL'd launch) discovered in the same investigation.

Scene generation: `tools/gen_sitl_obstacle_scenes.py`, extended 2026-08-24 to support a
`--seed`/`--n-scenes` calibration/test-style split (previously hardcoded to 8 scenes, one
fixed seed) — see that file's docstring.

### Tier 4b — Targeted mechanism stress test

Motivated directly by the 2026-08-24 finding: every authority level tested has the same
benign-looking QP-infeasible tick immediately after arming (pre-existing, not new), but the
**recovery** from that tick degrades with authority, and at `nom_accel_max`≥13 it can overshoot
into an unrecoverable divergence. A full Tier 4 episode (75s, needs obstacles + a goal) is a
very inefficient way to observe a transient that resolves (or doesn't) within the first ~2-3
seconds. Tier 4b isolates it directly:

- Obstacle-free scenes (`gen_sitl_obstacle_scenes.py --empty`), short window (~12s, not 75s)
  — the same SITL launch, no obstacles or goal-reaching logic needed, so a fresh PX4/Gazebo
  launch per cycle is still required but each cycle is far shorter.
- `tools/run_arming_transient_sweep.sh` sweeps `nom_accel_max`/`nom_pos_kp`/`nom_vel_kd`
  (editing the deploy YAML between batches, same pattern as the bisection scripts this
  session used) across a defined grid, running N cycles per authority level.
- `tools/parse_arming_transient.py` flags a cycle as "overshoot" if altitude exceeds a
  threshold or position departs the expected small radius within the observation window —
  does not require reaching a goal or avoiding obstacles, since there are none.

This produces a failure-rate-vs-authority curve instead of the 4 isolated spot-check data
points found manually on 2026-08-24 (8/10/12 clean, 13/14 failed) — cross-check any new run
against those known outcomes as a sanity check that the cheap proxy agrees with the expensive
full-episode result.

## Freezing a result

Once a config clears Tiers 0-4, freeze it into `results/frozen_<name>_<date>/MANIFEST.md`
following the existing convention (see `results/frozen_widerange_20260824/MANIFEST.md`,
`results/sitl_spotcheck_20260824/`): reproduction command, sha256-checksummed file table,
headline results table, an explicit **scope of validity** section (state plainly what was and
wasn't tested — sample sizes, scene distributions, what tier the claim rests on), and the
exact deploy YAML values the result assumes.

## Running everything (Tiers 0-3)

```
penn_gat_training/run_full_validation.sh
```

Runs Tier 0 (`colcon test`), Tier 1 (parity, if PENN enabled), Tier 2, Tier 3 in sequence and
reports pass/fail for each. Tiers 4/4b are **not** included — they are real-time-bound
(tens of minutes to hours) and must be invoked explicitly, never as a side effect of running
this script.
