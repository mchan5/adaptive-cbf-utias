"""Stage 2 of lcb_k selection: sweep on the BODY-RATE CLOSED-LOOP harness, using
CALIBRATION_SEEDS scenes only."""
import argparse
import csv
import os
import subprocess
import sys

LCB_KS = ["0.0", "0.25", "0.5", "0.75", "1.0", "1.5"]

_THIS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCENES = os.path.join(_THIS, "results", "closed_loop_calibration_20260823")
DEFAULT_WEIGHTS = os.path.join(_THIS, "..", "single_vehicle_cbf_rate_arc",
                               "nn_model", "checkpoint", "gat_penn_weights.pt")
DEFAULT_BIN = ("/home/matt/arc_2026/ros2_ws/install/single_vehicle_cbf_rate_arc/"
               "lib/single_vehicle_cbf_rate_arc/body_rate_closed_loop_diag")


def summarize(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r["arm"] == "adaptive(PENN)"]
    n = len(rows)
    hard = sum(1 for r in rows if float(r["min_h_physical"]) < 0)
    dead = sum(1 for r in rows if r["reached_goal"] == "0")
    mviol = sum(1 for r in rows if float(r["min_h_margin"]) < 0)
    per_scene = {(r["seed"], r["scene"]): (r["reached_goal"] == "0",
                                           float(r["min_h_margin"]) < 0) for r in rows}
    return {"n": n, "hard": hard, "deadlock": dead, "margin_viol": mviol, "per_scene": per_scene}


def mcnemar(a, b, index):
    """Discordant counts on the identical scene set. index 0=deadlock, 1=margin_viol."""
    keys = sorted(set(a["per_scene"]) & set(b["per_scene"]))
    only_a = sum(1 for k in keys if a["per_scene"][k][index] and not b["per_scene"][k][index])
    only_b = sum(1 for k in keys if b["per_scene"][k][index] and not a["per_scene"][k][index])
    return only_a, only_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=DEFAULT_SCENES)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--binary", default=DEFAULT_BIN)
    ap.add_argument("--out-dir", default=os.path.join(_THIS, "results",
                                                       "closed_loop_calibration_20260823", "sweep"))
    ap.add_argument("--skip-run", action="store_true",
                    help="Re-analyze existing CSVs in --out-dir without re-running episodes.")
    args = ap.parse_args()

    if not os.path.isdir(args.scenes):
        sys.exit(f"scene dir not found: {args.scenes}\n"
                 f"Run: CALIBRATION=1 python diag_gen_closed_loop_scenes.py")
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for k in LCB_KS:
        out_csv = os.path.join(args.out_dir, f"cal_k{k}.csv")
        if not args.skip_run:
            env = dict(os.environ, ADAPTIVE_ONLY="1", LCB_K=k)
            print(f"[lcb_k={k}] running ...", flush=True)
            with open(os.path.join(args.out_dir, f"run_k{k}.log"), "w") as logf:
                subprocess.run([args.binary, args.scenes, args.weights, out_csv],
                               env=env, stdout=logf, stderr=subprocess.STDOUT, check=True)
        results[k] = summarize(out_csv)

    print(f"\nCALIBRATION SEEDS ONLY -- adaptive arm, {results[LCB_KS[0]]['n']} episodes per value")
    print(f"{'lcb_k':>7} {'hard':>10} {'deadlock':>11} {'margin-viol':>13}")
    for k in LCB_KS:
        r = results[k]
        print(f"{k:>7} {r['hard']:>7}/{r['n']:<3} {r['deadlock']:>8}/{r['n']:<3} "
              f"{r['margin_viol']:>10}/{r['n']:<3}")

    eligible = [k for k in LCB_KS if results[k]["hard"] == 0]
    strict = min(eligible, key=lambda k: (results[k]["deadlock"], results[k]["margin_viol"]))
    print(f"\nStrict argmin (deadlock, then margin-viol) over 0-hard-collision values: lcb_k={strict}")

    print("\nPaired (McNemar) deadlock comparison against the strict pick -- a small")
    print("total difference is only meaningful if the discordant counts are lopsided:")
    for k in eligible:
        if k == strict:
            continue
        only_s, only_k = mcnemar(results[strict], results[k], 0)
        verdict = "indistinguishable" if abs(only_s - only_k) <= 2 else "separable"
        print(f"  k={strict} vs k={k}: {only_s} vs {only_k} discordant  -> {verdict}")

    print("\nSee this file's docstring for how the 2026-08-23 tie was broken "
          "(point-mass calibration evidence, no test-seed data).")


if __name__ == "__main__":
    main()
