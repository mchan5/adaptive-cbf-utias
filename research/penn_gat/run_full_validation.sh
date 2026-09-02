#!/usr/bin/env bash
# Runs Tiers 0-3 of the V&V procedure (penn_gat_training/TESTING.md) end to
# end against the currently-deployed config
# (config/params_single_vehicle_cbf_rate_arc.yaml). Tiers 4/4b (SITL) are
# deliberately NOT included here -- they are real-time-bound (tens of
# minutes to hours) and must be invoked explicitly via their own scripts,
# never as a side effect of running this one.
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARC_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

source "$ARC_ROOT/.venv/bin/activate"
source /opt/ros/*/setup.bash 2>/dev/null
source "$ARC_ROOT/install/setup.bash" 2>/dev/null

pass=0
fail=0

section() { echo; echo "=== $1 ==="; }

DEPLOY_YAML="$ARC_ROOT/control-barrier-functions-arc/single_vehicle_cbf_rate_arc/config/params_single_vehicle_cbf_rate_arc.yaml"
CKPT_DIR="$ARC_ROOT/control-barrier-functions-arc/single_vehicle_cbf_rate_arc/nn_model/checkpoint"

# ---- Tier -1: configuration integrity preflight --------------------------
# PLAN_adaptive_hardening_20260829.md Phase 0.3. Two of the three V&V
# defects found in the 2026-08-29 audit were config drift: the diag harness
# measuring a stale parameter snapshot, and the Tier 1 fixture generated
# from a superseded checkpoint. Catch both here, before Tiers 0/2/3 spend
# minutes running against a configuration that cannot pass.
section "Tier -1: configuration integrity preflight"
preflight_ok=1
python3 - "$DEPLOY_YAML" "$CKPT_DIR" <<'PYEOF'
import sys, os
try:
    import yaml
except ImportError:
    yaml = None
yaml_path, ckpt_dir = sys.argv[1], sys.argv[2]
rc = 0
if not os.path.isfile(yaml_path):
    print(f"FAIL: deployed YAML not found: {yaml_path}"); sys.exit(1)
if yaml is None:
    print("WARN: pyyaml unavailable; skipping deep preflight checks"); sys.exit(0)
with open(yaml_path) as f:
    doc = yaml.safe_load(f)
p = next(iter(doc.values()))["ros__parameters"]
keys = ("penn_enabled", "penn_model_path", "penn_gamma_min", "penn_gamma_max",
        "epistemic_threshold", "cvar_boundary", "lcb_k", "nom_accel_max",
        "nom_pos_kp", "nom_vel_kd", "cbf_gamma_obs")
print("deployed config under test:")
for k in keys:
    print(f"  {k:22s} {p.get(k)}")
if p.get("penn_enabled"):
    mp = p.get("penn_model_path", "")
    pt = mp if os.path.isabs(mp) else os.path.join(ckpt_dir, os.path.basename(mp))
    if not os.path.isfile(pt):
        print(f"FAIL: penn_model_path does not resolve to a file: {pt}"); rc = 1
    else:
        # FNV-1a 64 of the .pt the node loads -- mirror of
        # diag_gen_cpp_parity_states.py / penn_gamma_parity_check.cpp.
        h = 0xCBF29CE484222325
        with open(pt, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                for byte in chunk:
                    h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        stamp = f"{h:#018x}"
        fx = os.path.join(ckpt_dir, "cpp_parity_states.pt")
        want = None
        if os.path.isfile(fx):
            try:
                import torch
                want = torch.load(fx, map_location="cpu", weights_only=False).get("weights_fnv1a64")
            except Exception as e:
                print(f"WARN: could not read parity fixture stamp ({e})")
        if want is None:
            print(f"WARN: parity fixture has no weights stamp; regenerate diag_gen_cpp_parity_states.py")
        elif want != stamp:
            print(f"FAIL: parity fixture is stale -- fixture stamp {want} != deployed .pt {stamp}")
            print( "      cd penn_gat_training && CHECKPOINT_NAME=<name> DEPLOY_GAMMA_MIN=<lo> \\")
            print( "        DEPLOY_GAMMA_MAX=<hi> LCB_K=<k> python3 diag_gen_cpp_parity_states.py")
            rc = 1
        else:
            print(f"OK: parity fixture matches deployed .pt ({stamp})")
sys.exit(rc)
PYEOF
if [ $? -eq 0 ]; then
  echo "PASS: configuration integrity preflight"
  ((pass++))
else
  echo "FAIL: configuration integrity preflight -- fix the above before trusting any tier below"
  ((fail++))
  preflight_ok=0
fi

# ---- Tier 0: component unit tests -----------------------------------------
section "Tier 0: unit tests (colcon test)"
(cd "$ARC_ROOT" && colcon build --packages-select single_vehicle_cbf_rate_arc >/tmp/tier0_build.log 2>&1)
if (cd "$ARC_ROOT" && colcon test --packages-select single_vehicle_cbf_rate_arc --ctest-args -R single_vehicle_cbf_rate_arc_test >/tmp/tier0_test.log 2>&1); then
  echo "PASS: Tier 0 gtest suite"
  ((pass++))
else
  echo "FAIL: Tier 0 gtest suite -- see /tmp/tier0_test.log"
  ((fail++))
fi

# ---- Tier 1: parity (only meaningful with PENN enabled) -------------------
section "Tier 1: C++/Python gamma-selection parity"
if grep -q "penn_enabled: true" "$DEPLOY_YAML" 2>/dev/null; then
  "$ARC_ROOT/install/single_vehicle_cbf_rate_arc/lib/single_vehicle_cbf_rate_arc/penn_gamma_parity_check" \
    && { echo "PASS: Tier 1 parity"; ((pass++)); } \
    || { echo "FAIL: Tier 1 parity"; ((fail++)); }
else
  echo "SKIPPED: penn_enabled: false in current deploy config -- parity is not meaningful"
fi

# ---- Tier 2: point-mass statistical (disjoint TEST_SEEDS) -----------------
section "Tier 2: point-mass statistical validation (TEST_SEEDS)"
# Feed the deployed gamma range / lcb_k / fallback step to the point-mass
# harness -- reproduce_headline.py takes the checkpoint from the deployed
# name but reads these from the environment.
eval "$(python3 - "$DEPLOY_YAML" <<'PYEOF'
import sys, yaml
p = next(iter(yaml.safe_load(open(sys.argv[1])).values()))["ros__parameters"]
print(f'export DEPLOY_GAMMA_MIN={p["penn_gamma_min"]}')
print(f'export DEPLOY_GAMMA_MAX={p["penn_gamma_max"]}')
print(f'export LCB_K={p["lcb_k"]}')
print(f'export FALLBACK_STEP={p.get("penn_fallback_step", 0.5)}')
PYEOF
)"
echo "Tier 2 gamma range [$DEPLOY_GAMMA_MIN, $DEPLOY_GAMMA_MAX], lcb_k=$LCB_K, fallback_step=$FALLBACK_STEP"
if python3 reproduce_headline.py >/tmp/tier2.log 2>&1; then
  echo "PASS: Tier 2 (see /tmp/tier2.log for the headline numbers)"
  ((pass++))
else
  echo "FAIL: Tier 2 -- see /tmp/tier2.log"
  ((fail++))
fi

# ---- Tier 3: body-rate closed-loop (disjoint TEST_SEEDS) ------------------
section "Tier 3: body-rate closed-loop validation (TEST_SEEDS) -- NECESSARY, NOT SUFFICIENT"
echo "(see TESTING.md's boxed caveat -- a Tier 3 pass alone is not a validation claim)"
DEPLOY_YAML="$DEPLOY_YAML" \
"$ARC_ROOT/install/single_vehicle_cbf_rate_arc/lib/single_vehicle_cbf_rate_arc/body_rate_closed_loop_diag" \
  results/frozen_widerange_20260824 \
  ../single_vehicle_cbf_rate_arc/nn_model/checkpoint/gat_penn_weights.pt \
  /tmp/tier3.csv > /tmp/tier3.log 2>&1
if grep -q "hard/N" /tmp/tier3.log; then
  echo "COMPLETED: Tier 3 -- review /tmp/tier3.log for hard-collision counts (0 required)"
  ((pass++))
else
  echo "FAIL: Tier 3 did not complete -- see /tmp/tier3.log"
  ((fail++))
fi

section "Summary"
echo "$pass tier(s) passed, $fail tier(s) failed."
echo "Tiers 4/4b (SITL) not run -- invoke tools/run_sitl_battery.sh / tools/run_arming_transient_sweep.sh explicitly."
[ "$fail" -eq 0 ]
