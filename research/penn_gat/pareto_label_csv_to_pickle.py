"""
Phase 2b of the Pareto-reframe plan (2026-08-25): converts the CSV rows from
body_rate_pareto_label_export (per-decision-point, per-candidate-gamma
state + label, from the corrected within-episode oracle) into the exact
same pickle schema drone_data_generation.py::worker() already produces --
a list of {"graph_data": {x, edge_index, edge_attr, y, gamma}} dicts -- via
the SAME GATModule3D.create_graph() call worker() uses, so train_drone.py
needs zero code changes to consume this data. Only the label SOURCE differs
(real body-rate closed-loop oracle rollouts of states an actual episode
visits, vs. the old point-mass carrier-trajectory sampling); the schema and
downstream training/calibration code are untouched.

Usage:
    python pareto_label_csv_to_pickle.py <in.csv> <out.pkl>
"""
import csv
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nn_model"))
from gat_3d import GATModule3D  # noqa: E402

DRONE_RADIUS = 0.25


def parse_obstacles(field):
    """field: "ox,oy,oz,r;ox,oy,oz,r;..." -> [[ox,oy,oz,r,0,0,0], ...]
    (trailing zeros match worker()'s obs_list convention: obstacles are
    treated as static, vx=vy=vz=0)."""
    if not field:
        return []
    out = []
    for chunk in field.split(";"):
        ox, oy, oz, r = (float(x) for x in chunk.split(","))
        out.append([ox, oy, oz, r, 0.0, 0.0, 0.0])
    return out


def main(csv_path, out_path):
    module = GATModule3D(robot_radius=DRONE_RADIUS)
    results = []
    skipped = 0

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            obstacles = parse_obstacles(row["obstacles"])
            if not obstacles:
                skipped += 1
                continue
            robot_state = [float(row["px"]), float(row["py"]), float(row["pz"]),
                            float(row["vx"]), float(row["vy"]), float(row["vz"])]
            goal = [float(row["gx"]), float(row["gy"]), float(row["gz"])]
            progress_deficit = float(row["progress_deficit"])
            min_h_horizon = float(row["min_h_horizon"])
            gamma = float(row["gamma"])

            graph = module.create_graph(
                robot=robot_state, obstacles=obstacles, goal=goal,
                deadlock=progress_deficit, risk=min_h_horizon,
            )
            graph.gamma = graph.gamma if hasattr(graph, "gamma") else None
            import torch
            graph.gamma = torch.tensor([[gamma]], dtype=torch.float)

            results.append({
                "graph_data": {
                    "x": graph.x.cpu().numpy(),
                    "edge_index": graph.edge_index.cpu().numpy(),
                    "edge_attr": graph.edge_attr.cpu().numpy(),
                    "y": graph.y.cpu().numpy(),
                    "gamma": graph.gamma.cpu().numpy(),
                },
                # episode_id (2026-08-26, PLAN_adaptive_recovery §1.3):
                # (seed, scene) uniquely identifies the source episode this
                # decision point was drawn from -- the CSV already carries
                # it, this just doesn't drop it on the way into the pickle.
                # Extra dict key is a no-op for every existing reader
                # (train_drone.py/cccp_calibrate_drone.py only index
                # item["graph_data"]); lets a future trajectory-level
                # conformal calibration (Q_i = max_t D(X_t) grouped by
                # episode_id) be computed without regenerating data again.
                "episode_id": f"{row['seed']}_{row['scene']}",
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Wrote {len(results):,} samples to {out_path} ({skipped} rows skipped, no obstacles)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <in.csv> <out.pkl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
