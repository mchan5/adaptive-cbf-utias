#!/usr/bin/env python3
"""One-time export: pull the raw tensors the C++ PennGammaSelector needs out of the validated
Quadrotor3D_gat.pth checkpoint, and save them as a flat {str: Tensor} dict"""
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# CHECKPOINT_PATH/OUTPUT_PATH env overrides added 2026-08-23, matching train_drone.py's pattern, so
# a candidate checkpoint can be exported to a scratch location for parity-check/campaign testing …
_CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH") or os.path.join(
    _THIS_DIR, "nn_model", "checkpoint", "Quadrotor3D_gat.pth")
_OUTPUT_PATH = os.environ.get("EXPORT_OUTPUT_PATH") or os.path.join(
    _THIS_DIR, "..", "single_vehicle_cbf_rate_arc", "nn_model",
    "checkpoint", "gat_penn_weights.pt")


def main():
    # BUG (found 2026-08-22): this used to check `if _THIS_DIR not in sys.path` before inserting
    # `_THIS_DIR/nn_model` -- checking one path and inserting a different one.
    nn_model_dir = os.path.join(_THIS_DIR, "nn_model")
    if nn_model_dir not in sys.path:
        sys.path.insert(0, nn_model_dir)

    from gat_3d import GATModule3D
    from penn.nn_gat_iccbf_predict import ProbabilisticEnsembleGAT

    device = "cpu"
    gat_module = GATModule3D(robot_radius=0.25, device=device).to(device)
    penn_gat = ProbabilisticEnsembleGAT(
        gat_module.gat, n_output=2, n_hidden=40, n_ensemble=3,
        gamma_dim=1, device=device, lr=0.0003,
    ).to(device)
    penn_gat.load_model(_CHECKPOINT_PATH)
    penn_gat.eval()

    gat_net = penn_gat.gat_network  # GATGraphNetwork3D
    ens = penn_gat.model            # EnsembleStochasticLinear

    weights = {}

    def add_sequential(name, seq):
        # seq is nn.Sequential(Linear, ReLU, Linear) -- indices 0 and 2 are
        # the Linear layers, matching gat_3d.py's psi1/psi2/psi3 definitions.
        weights[f"{name}.0.weight"] = seq[0].weight.detach().clone()
        weights[f"{name}.0.bias"] = seq[0].bias.detach().clone()
        weights[f"{name}.2.weight"] = seq[2].weight.detach().clone()
        weights[f"{name}.2.bias"] = seq[2].bias.detach().clone()

    add_sequential("psi1", gat_net.psi1)
    add_sequential("psi2", gat_net.psi2)
    add_sequential("psi3", gat_net.psi3)

    for lin_name in ("lin1", "lin2", "lin3", "lin4", "lin5"):
        lin = getattr(ens, lin_name)
        weights[f"{lin_name}.weights"] = lin.weights.detach().clone()
        weights[f"{lin_name}.biases"] = lin.biases.detach().clone()

    weights["gamma_max"] = torch.tensor(float(penn_gat.gamma_max))
    weights["log_std_min"] = torch.tensor(float(ens.log_std_min))
    weights["log_std_max"] = torch.tensor(float(ens.log_std_max))
    weights["ensemble_size"] = torch.tensor(int(ens.ensemble_size))
    weights["n_output"] = torch.tensor(int(ens.n_output))
    weights["zij_dim"] = torch.tensor(int(gat_net.zij_dim))
    weights["gamma_dim"] = torch.tensor(int(gat_net.gamma_dim))
    weights["robot_radius"] = torch.tensor(float(gat_module.robot_radius))

    os.makedirs(os.path.dirname(_OUTPUT_PATH), exist_ok=True)

    torch.save(weights, _OUTPUT_PATH)
    print(f"Wrote {len(weights)} tensors to {_OUTPUT_PATH}")
    for k, v in weights.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
