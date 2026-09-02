#!/usr/bin/env bash
# Curate 20 SITL-clean hardware scenes (12 passable + 8 reroute). For each pool candidate: run
# fixed(g5.0,penn off) then adaptive(penn on) through SITL, judge both.
source "$(dirname "$0")/lib.sh"
CUR="$(dirname "$0")"
POOL="$CUR/pool"
FINAL="$REPO/penn_gat_training/results/hardware_scenes_final_20260831"
SCENES_OUT="$FINAL/scenes"
RUNS="$FINAL/sitl_runs"
WINDOW=${WINDOW:-95}
NEED_PASS=12
NEED_RE=8
mkdir -p "$SCENES_OUT" "$RUNS"
echo $$ > "$FINAL/curate.pid"

cp "$SRC_YAML" "$YAML"
trap 'cp "$SRC_YAML" "$YAML"; sitl_cleanup; echo "[restore] install yaml <- source"' EXIT INT TERM
sitl_env
set_authority 8 4 4
# goal-arrival tolerance 0.3 -> 0.6 (stops the at-goal stall-escape wander;
# matches judge.py GOAL_TOL). Stall-escape left ON for deadlock recovery.
python3 - "$YAML" <<'PY'
import re, sys
y = sys.argv[1]; t = open(y).read()
t = re.sub(r'(\n\s*goal_arrival_tolerance:)\s*[0-9.]+', r'\g<1> 0.6', t)
open(y, 'w').write(t)
PY

MANIFEST="$FINAL/MANIFEST.csv"
HDR="final_id,class,n_obs,cand,fixed_strong,fixed_reached,fixed_tgoal,fixed_minclear,fixed_veer,adaptive_strong,adaptive_reached,adaptive_tgoal,adaptive_minclear,adaptive_veer"
[ -s "$MANIFEST" ] || echo "$HDR" > "$MANIFEST"
LOG="$FINAL/curate.log"
echo "=== CURATE start $(date) ===" | tee -a "$LOG"

run_sitl() {  # $1 scene json  $2 out log  ; penn/gamma already set
  [ -s "$2" ] && grep -q 'pos=\[' "$2" && return 0
  sitl_cleanup; sleep 2
  timeout "${WINDOW}s" ros2 launch single_vehicle_cbf_rate_arc cbf_rate_arc_sitl_test_launch.py \
    launch_rviz:=false obstacle_file:="$1" \
    px4_src_dir:="$PX4_SRC" micro_xrce_agent_src_dir:="$XRCE_SRC" > "$2" 2>&1
  return 0
}

# resume-safe: recount from what is already accepted on disk
n_pass=$(ls "$SCENES_OUT"/scene_0[0-9].json "$SCENES_OUT"/scene_1[01].json 2>/dev/null | wc -l)
n_re=$(ls "$SCENES_OUT"/scene_1[2-9].json 2>/dev/null | wc -l)
echo "resume: passable=$n_pass reroute=$n_re already accepted" | tee -a "$LOG"

for cand in $(ls "$POOL"/cand_*.json | sort); do
  cbase=$(basename "$cand" .json)
  grep -q ",$cbase," "$MANIFEST" && { echo "  $cbase already in manifest, skip"; continue; }
  cls=$(grep ",$cbase," /dev/null 2>/dev/null; awk -F, -v c="$cbase" '$1==c{print $2}' "$POOL/pool_manifest.csv")
  nobs=$(awk -F, -v c="$cbase" '$1==c{print $3}' "$POOL/pool_manifest.csv")
  [ "$cls" = "passable" ] && [ "$n_pass" -ge "$NEED_PASS" ] && continue
  [ "$cls" = "reroute" ]  && [ "$n_re"   -ge "$NEED_RE"   ] && continue
  [ "$n_pass" -ge "$NEED_PASS" ] && [ "$n_re" -ge "$NEED_RE" ] && break

  echo "--- $cbase ($cls n=$nobs)  $(date +%T) ---" | tee -a "$LOG"
  set_penn false; set_gamma 5.0
  run_sitl "$cand" "$RUNS/${cbase}_fixed.log"
  fj=$(python3 "$CUR/judge.py" "$RUNS/${cbase}_fixed.log" "$cand"); frc=$?
  echo "  fixed:    $fj" | tee -a "$LOG"

  set_penn true
  run_sitl "$cand" "$RUNS/${cbase}_adaptive.log"
  aj=$(python3 "$CUR/judge.py" "$RUNS/${cbase}_adaptive.log" "$cand"); arc=$?
  echo "  adaptive: $aj" | tee -a "$LOG"

  get() { echo "$1" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('$2'))"; }
  if [ "$frc" -eq 0 ] && [ "$arc" -eq 0 ]; then
    if [ "$cls" = "passable" ]; then fid=$n_pass; pre="p"; n_pass=$((n_pass+1));
    else fid=$((12 + n_re)); pre="r"; n_re=$((n_re+1)); fi
    fid2=$(printf "%02d" "$fid")
    cp "$cand" "$SCENES_OUT/scene_${fid2}.json"
    echo "$fid2,$cls,$nobs,$cbase,1,$(get "$fj" reached),$(get "$fj" t_goal),$(get "$fj" min_clear),$(get "$fj" max_absx_corridor),1,$(get "$aj" reached),$(get "$aj" t_goal),$(get "$aj" min_clear),$(get "$aj" max_absx_corridor)" >> "$MANIFEST"
    echo "  ACCEPT -> scene_${fid2}  (passable $n_pass/$NEED_PASS, reroute $n_re/$NEED_RE)" | tee -a "$LOG"
  else
    echo "  reject (fixed rc=$frc adaptive rc=$arc)" | tee -a "$LOG"
  fi
done

cp "$SRC_YAML" "$YAML"; sitl_cleanup
echo "=== CURATE done $(date)  accepted passable=$n_pass reroute=$n_re ===" | tee -a "$LOG"
if [ "$n_pass" -ge "$NEED_PASS" ] && [ "$n_re" -ge "$NEED_RE" ]; then
  ( cd "$REPO/penn_gat_training" && python3 "$CUR/finalize.py" "$FINAL" ) | tee -a "$LOG"
  echo "CURATE_COMPLETE" | tee -a "$LOG"
else
  echo "CURATE_SHORT need more pool candidates (passable $n_pass/12 reroute $n_re/8)" | tee -a "$LOG"
fi
