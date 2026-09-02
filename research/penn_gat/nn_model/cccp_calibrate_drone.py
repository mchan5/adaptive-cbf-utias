"""Epistemic (JRD) threshold calibration for Quadrotor3D_gat."""

import json
import os
import sys
import pickle

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

sys.path.insert(0, os.path.dirname(__file__))
from gat_3d import GATModule3D
from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT

# ── Config ────────────────────────────────────────────────────────────────────

# MODEL_NAME/PICKLE env overrides added 2026-08-23, matching train_drone.py's
# pattern, to calibrate a candidate checkpoint without touching the live one.
MODEL_NAME = os.environ.get("MODEL_NAME", "Quadrotor3D_gat")
PICKLE     = os.environ.get("PICKLE_FILE", "../data/drone_data_100000.pkl")
MODEL_PATH = f"checkpoint/{MODEL_NAME}.pth"
OUT_JSON   = f"checkpoint/cccp_threshold_{MODEL_NAME}.json"
ALPHA_CAL  = 0.95
GAMMA_DIM  = 1
BATCH_SIZE = 512
device     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_graph_dataset(pickle_file):
    with open(pickle_file, 'rb') as f:
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


def collect_jrd(penn_gat, data_list, batch_size=512):
    loader = GeoDataLoader(data_list, batch_size=batch_size, shuffle=False)
    penn_gat.gat_network.eval()
    penn_gat.model.eval()
    jrd_vals = []
    with torch.no_grad():
        for batch in loader:
            x     = batch.x.to(device)
            ei    = batch.edge_index.to(device)
            ea    = batch.edge_attr.to(device)
            bidx  = batch.batch.to(device)
            gamma = batch.gamma.view(-1, GAMMA_DIM).to(device) / penn_gat.gamma_max
            emb   = penn_gat.gat_network.extract_robot_embedding(x, ei, ea, bidx)
            vel   = penn_gat._robot_vel_feature(x, bidx)
            X_in  = torch.cat([emb, vel, gamma], dim=1)
            ens   = penn_gat.model(X_in)
            jrd   = ens[-1][:, 0].cpu().numpy()
            jrd_vals.extend(jrd.tolist())
    return np.array(jrd_vals, dtype=np.float64)


def collect_aleatoric_boundary(penn_gat, data_list, batch_size=512,
                                alpha=0.95,
                                safe_min_h_threshold=0.0,
                                safe_deadlock_threshold=float("inf")):
    """Lower-(1-alpha)-percentile of ensemble mean_mu (risk-head output) over samples where
    true_min_h >= safe_min_h_threshold."""
    loader = GeoDataLoader(data_list, batch_size=batch_size, shuffle=False)
    penn_gat.gat_network.eval()
    penn_gat.model.eval()
    mean_mus, true_min_hs, deadlock_vals = [], [], []

    with torch.no_grad():
        for batch in loader:
            x     = batch.x.to(device)
            ei    = batch.edge_index.to(device)
            ea    = batch.edge_attr.to(device)
            bidx  = batch.batch.to(device)
            gamma = batch.gamma.view(-1, GAMMA_DIM).to(device) / penn_gat.gamma_max
            emb   = penn_gat.gat_network.extract_robot_embedding(x, ei, ea, bidx)
            vel   = penn_gat._robot_vel_feature(x, bidx)
            X_in  = torch.cat([emb, vel, gamma], dim=1)
            ens   = penn_gat.model(X_in)

            mus = [mu_ls[0][:, 1].cpu().numpy() for mu_ls in ens[:-1]]
            mean_mus.extend(np.mean(np.stack(mus, axis=0), axis=0).tolist())

            if batch.y is not None:
                y = batch.y.view(-1, 2)
                true_min_hs.extend(y[:, 1].cpu().numpy().tolist())
                deadlock_vals.extend(y[:, 0].cpu().numpy().tolist())

    mean_mus      = np.array(mean_mus)
    true_min_hs   = np.array(true_min_hs)
    deadlock_vals = np.array(deadlock_vals)

    safe_mask = (true_min_hs >= safe_min_h_threshold) & (deadlock_vals < safe_deadlock_threshold)
    n_safe = int(safe_mask.sum())
    print(f"[aleatoric-cal] Safe samples: {n_safe:,} / {len(safe_mask):,} "
          f"({100*n_safe/len(safe_mask):.1f}%)")
    if n_safe == 0:
        return float(np.quantile(mean_mus, 1 - alpha))
    safe_mus = mean_mus[safe_mask]
    boundary = float(np.quantile(safe_mus, 1 - alpha))
    print(f"[aleatoric-cal] cvar_boundary (lower bound) @ alpha={alpha}: {boundary:.4f}")
    return boundary


def collect_risk_noise_floor(penn_gat, data_list, batch_size=512):
    """Mean predicted std-dev of the risk head across the ensemble."""
    loader = GeoDataLoader(data_list, batch_size=batch_size, shuffle=False)
    penn_gat.gat_network.eval()
    penn_gat.model.eval()
    sigmas = []
    with torch.no_grad():
        for batch in loader:
            x, ei, ea = batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device)
            bidx = batch.batch.to(device)
            gamma = batch.gamma.view(-1, GAMMA_DIM).to(device) / penn_gat.gamma_max
            emb = penn_gat.gat_network.extract_robot_embedding(x, ei, ea, bidx)
            vel = penn_gat._robot_vel_feature(x, bidx)
            X_in = torch.cat([emb, vel, gamma], dim=1)
            ens = penn_gat.model(X_in)
            for mu, log_std in ens[:-1]:
                sigmas.extend(torch.exp(log_std[:, 1]).cpu().numpy().tolist())
    sigmas = np.array(sigmas)
    eps = float(np.mean(sigmas))
    print(f"[risk-noise] risk-head sigma: mean={eps:.4f}  median={np.median(sigmas):.4f}  "
          f"p90={np.percentile(sigmas,90):.4f}")
    return eps


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {PICKLE} ...")
    data_list = load_graph_dataset(PICKLE)
    print(f"Loaded {len(data_list):,} graphs.")

    gat_mod  = GATModule3D(device=device).to(device)
    penn_gat = ProbabilisticEnsembleGAT(
        gat_mod.gat, n_output=2, n_hidden=40, n_ensemble=3,
        gamma_dim=GAMMA_DIM, device=device, lr=0.0003,
    ).to(device)
    penn_gat.load_model(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    # Epistemic threshold
    print("\nCollecting JRD values ...")
    jrd_vals  = collect_jrd(penn_gat, data_list, batch_size=BATCH_SIZE)
    threshold = float(np.quantile(jrd_vals, ALPHA_CAL, interpolation="higher"))

    print(f"\nJRD  min={jrd_vals.min():.6f}  mean={jrd_vals.mean():.6f}  "
          f"p95={threshold:.6f}  max={jrd_vals.max():.6f}")
    print(f"raw_epistemic_threshold: {threshold:.6f}")

    # Aleatoric boundary (use last 20% as validation, same split as training)
    n_val = int(len(data_list) * 0.2)
    val_list = data_list[-n_val:]
    print(f"\nCollecting aleatoric boundary on {len(val_list):,} val samples ...")
    cvar_bnd = collect_aleatoric_boundary(penn_gat, val_list, batch_size=BATCH_SIZE)

    print(f"\nCollecting risk noise floor on {len(val_list):,} val samples ...")
    risk_tie_eps = collect_risk_noise_floor(penn_gat, val_list, batch_size=BATCH_SIZE)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    result = {
        "model": MODEL_NAME,
        "alpha_cal": ALPHA_CAL,
        "raw_epistemic_threshold": threshold,
        "cvar_boundary": cvar_bnd,
        "risk_tie_eps": risk_tie_eps,
        "jrd_stats": {
            "min":       float(jrd_vals.min()),
            "max":       float(jrd_vals.max()),
            "mean":      float(jrd_vals.mean()),
            "std":       float(jrd_vals.std()),
            "n_samples": int(jrd_vals.size),
        },
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved → {OUT_JSON}")
    print("\n=== Paste into inference node config ===")
    print(f'  raw_epistemic_threshold: {threshold:.6f}')
    print(f'  cvar_boundary:           {cvar_bnd:.4f}  (aleatoric gate, LOWER bound on predicted min_h as of 2026-08-20)')
    print(f'  risk_tie_eps:            {risk_tie_eps:.4f}  (vestigial, unused by the gates-then-argmin-performance rule)')
