"""PENN+GAT training for the quadrotor CBF. Run from the nn_model/ directory: python
train_drone.py Output: checkpoint/Quadrotor3D_gat.pth"""

import os
import sys
import time
import random
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
from gat_3d import GATModule3D
from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT

# ── Config ────────────────────────────────────────────────────────────────────

# PICKLE_FILE/MODEL_NAME env overrides added 2026-08-23 for the range-widening retrain, so a
# candidate checkpoint can be trained and gated (aleatoric calibration, headline re-verification, …
PICKLE_FILE  = os.environ.get("PICKLE_FILE", "../data/drone_data_100000.pkl")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Quadrotor3D_gat")
model_path   = f"checkpoint/{MODEL_NAME}.pth"
resume_path  = f"checkpoint/{MODEL_NAME}_resume.pth"

GAMMA_DIM   = 1      # only alpha_obs
LR          = 0.0003
BATCHSIZE   = 256
# EPOCH env override added 2026-08-23 to support a short smoke retrain (~20-30 epochs, this
# project's established pattern for catching a degenerate/broken retrain cheaply before committing …
EPOCH       = int(os.environ.get("EPOCH", 300))
N_HIDDEN    = 40
N_ENSEMBLE  = 3
# [progress_deficit, min_h_horizon], both receding-horizon quantities.
N_OUTPUT    = 2
ACTIVATION  = 'relu'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# ── Data loading ──────────────────────────────────────────────────────────────

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
        g.y = (torch.tensor(gd["y"], dtype=torch.float)
               if gd.get("y") is not None else torch.zeros(1, 2))
        data_list.append(g)
    return data_list


# ── Training ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Loading {PICKLE_FILE} ...")
    graph_list = load_graph_dataset(PICKLE_FILE)
    print(f"Loaded {len(graph_list):,} graphs.")

    n_tot = len(graph_list)
    n_trn = int(0.8 * n_tot)
    train_g, test_g = torch.utils.data.random_split(graph_list, [n_trn, n_tot - n_trn])

    gat_module  = GATModule3D(device=device).to(device)
    gat_network = gat_module.gat
    penn_gat = ProbabilisticEnsembleGAT(
        gat_network, N_OUTPUT, N_HIDDEN, N_ENSEMBLE,
        GAMMA_DIM, device, LR, ACTIVATION,
    ).to(device)

    os.makedirs('checkpoint', exist_ok=True)
    best_rmse     = 1e9
    start_epoch   = 0
    prior_elapsed = 0.0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        penn_gat.optimizer, T_max=EPOCH, eta_min=LR * 0.01
    )

    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path} ...")
        ckpt = torch.load(resume_path, map_location=device)
        penn_gat.load_state_dict(ckpt['model_state_dict'])
        penn_gat.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch   = ckpt['epoch'] + 1
        best_rmse     = ckpt['best_test_rmse']
        prior_elapsed = ckpt.get('elapsed_seconds', 0.0)
        print(f"  Epoch {start_epoch}, best RMSE {best_rmse:.4f}")

    train_loader = GeoDataLoader(train_g, batch_size=BATCHSIZE, shuffle=True)
    test_loader  = GeoDataLoader(test_g,  batch_size=BATCHSIZE, shuffle=False)

    t0 = time.time()
    for epoch in range(start_epoch, EPOCH):
        train_loss          = penn_gat.train_epoch(train_loader, epoch)
        test_loss, test_rmse = penn_gat.test(test_loader,  epoch)
        scheduler.step()

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            print("  → new best, saving ...")
            torch.save(penn_gat.state_dict(), model_path)

        torch.save({
            'epoch':                epoch,
            'model_state_dict':     penn_gat.state_dict(),
            'optimizer_state_dict': penn_gat.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_test_rmse':       best_rmse,
            'elapsed_seconds':      prior_elapsed + (time.time() - t0),
        }, resume_path)

    total = prior_elapsed + (time.time() - t0)
    print(f"Training complete — {total / 60:.1f} min")
    if os.path.exists(resume_path):
        os.remove(resume_path)
