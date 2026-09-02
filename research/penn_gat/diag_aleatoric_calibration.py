"""Is the PENN's aleatoric (per-member predictive variance) head actually estimating anything, or
is it degenerate? Background: until 2026-08-22 it was degenerate."""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nn_model"))

from gat_3d import GATModule3D  # noqa: E402
from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT  # noqa: E402

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__),
                            "nn_model/checkpoint/Quadrotor3D_gat.pth")
PICKLE = os.path.join(os.path.dirname(__file__), "data/drone_data_100000.pkl")
GAMMA_DIM = 1
BATCH_SIZE = 512
RISK_CHANNEL = 1  # y[:,1] = min_h_horizon


def load_graph_dataset(pickle_file):
    with open(pickle_file, "rb") as f:
        results = pickle.load(f)
    data_list = []
    for item in results:
        gd = item["graph_data"]
        g = Data(
            x=torch.tensor(gd["x"], dtype=torch.float),
            edge_index=torch.tensor(gd["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(gd["edge_attr"], dtype=torch.float),
        )
        if gd.get("gamma") is not None:
            g.gamma = torch.tensor(gd["gamma"], dtype=torch.float)
        if gd.get("y") is not None:
            g.y = torch.tensor(gd["y"], dtype=torch.float)
        data_list.append(g)
    return data_list


def collect(penn_gat, data_list, device):
    """Returns (mean_sigma, abs_residual) per validation sample, both for the risk channel."""
    loader = GeoDataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    penn_gat.gat_network.eval()
    penn_gat.model.eval()
    sigmas, residuals = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch.x.to(device)
            ei = batch.edge_index.to(device)
            ea = batch.edge_attr.to(device)
            bidx = batch.batch.to(device)
            gamma = batch.gamma.view(-1, GAMMA_DIM).to(device) / penn_gat.gamma_max
            emb = penn_gat.gat_network.extract_robot_embedding(x, ei, ea, bidx)
            vel = penn_gat._robot_vel_feature(x, bidx)
            ens = penn_gat.model(torch.cat([emb, vel, gamma], dim=1))

            mus = np.stack([mu[:, RISK_CHANNEL].cpu().numpy() for mu, _ in ens[:-1]], axis=0)
            sig = np.stack([torch.exp(ls[:, RISK_CHANNEL]).cpu().numpy() for _, ls in ens[:-1]],
                           axis=0)
            y_true = batch.y.view(-1, 2)[:, RISK_CHANNEL].cpu().numpy()
            sigmas.extend(sig.mean(axis=0).tolist())
            residuals.extend(np.abs(mus.mean(axis=0) - y_true).tolist())
    return np.array(sigmas), np.array(residuals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--log-std-max", type=float, default=None,
                    help="Override the model's log_std ceiling. Use -2 to reproduce "
                          "the pre-2026-08-22 degenerate configuration.")
    ap.add_argument("--pickle", default=PICKLE,
                    help="Validation dataset (2026-08-23: override to check a "
                         "checkpoint against the distribution it was actually "
                         "trained on, e.g. data/drone_data_widerange_100000.pkl "
                         "for the range-widening retrain -- the default only "
                         "covers the original (0.2,3.5) gamma range.")
    args = ap.parse_args()
    device = "cpu"

    full = load_graph_dataset(args.pickle)
    n_val = int(len(full) * 0.2)
    val = full[-n_val:]  # same trailing-20% split train_drone.py holds out

    gat_mod = GATModule3D(device=device).to(device)
    penn_gat = ProbabilisticEnsembleGAT(
        gat_mod.gat, n_output=2, n_hidden=40, n_ensemble=3,
        gamma_dim=GAMMA_DIM, device=device, lr=0.0003,
    ).to(device)
    penn_gat.load_model(args.checkpoint)
    if args.log_std_max is not None:
        penn_gat.model.log_std_max = args.log_std_max
    ceiling = float(np.exp(penn_gat.model.log_std_max))
    print(f"checkpoint: {args.checkpoint}")
    print(f"log_std clamp: [{penn_gat.model.log_std_min}, {penn_gat.model.log_std_max}] "
          f"-> sigma ceiling {ceiling:.4f}"
          + ("  [OVERRIDDEN]" if args.log_std_max is not None else ""))
    print(f"val samples: {len(val):,}\n")

    sigma, resid = collect(penn_gat, val, device)

    print("--- 1. SPREAD (a degenerate head is near-constant here) ---")
    print(f"  sigma mean={sigma.mean():.4f}  median={np.median(sigma):.4f}  std={sigma.std():.4f}")
    print(f"  p10={np.percentile(sigma,10):.4f}  p90={np.percentile(sigma,90):.4f}  "
          f"max={sigma.max():.4f}")
    print(f"  distinct values (4dp): {len(np.unique(np.round(sigma, 4))):,}")

    # The actual signature of the 2026-08-22 failure: sigma pinned AT the ceiling.
    sat_frac = float((sigma >= 0.99 * ceiling).mean())
    print(f"\n  SATURATION: {100*sat_frac:.1f}% of samples pinned at the sigma ceiling "
          f"({ceiling:.4f})")

    print("\n--- 2. CALIBRATION (does sigma track the model's own error?) ---")
    pearson = float(np.corrcoef(sigma, resid)[0, 1])
    rank_s = np.argsort(np.argsort(sigma))
    rank_r = np.argsort(np.argsort(resid))
    spearman = float(np.corrcoef(rank_s, rank_r)[0, 1])
    print(f"  Pearson  corr(sigma, |residual|): {pearson:.4f}")
    print(f"  Spearman corr(sigma, |residual|): {spearman:.4f}")

    print("\n  decile | mean sigma | mean |residual|   (should rise monotonically)")
    order = np.argsort(sigma)
    n = len(order)
    prev = -np.inf
    monotonic = True
    for d in range(10):
        idx = order[int(d * n / 10):int((d + 1) * n / 10)]
        mr = resid[idx].mean()
        if mr < prev:
            monotonic = False
        prev = mr
        print(f"    {d}    | {sigma[idx].mean():10.4f} | {mr:14.4f}")

    print(f"\n  monotonic across deciles: {monotonic}")

    reasons = []
    if sat_frac > 0.5:
        reasons.append(f"{100*sat_frac:.0f}% of samples pinned at the sigma ceiling "
                       f"(raise log_std_max -- this was the 2026-08-22 bug)")
    if sigma.std() < 1e-3:
        reasons.append("sigma is effectively constant across the validation set")
    if abs(pearson) < 0.4:
        reasons.append(f"sigma barely tracks the model's own error (Pearson {pearson:.2f} < 0.4)")

    print()
    if reasons:
        print("VERDICT: DEGENERATE / POORLY CALIBRATED")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("VERDICT: OK -- head is informative and calibrated")
    return 1 if reasons else 0


if __name__ == "__main__":
    sys.exit(main())
