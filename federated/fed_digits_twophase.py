#!/usr/bin/env python
"""
Two-phase federated learning on the Digits benchmark.

Phase 1  – Local warm-up ("single training")
------------------------------------------------
Each client trains an ordinary **DigitModel** on its own dataset for a
user-defined number of epochs (default: 150). This yields a strong local
expert that captures domain-specific knowledge without any cross-client
interference.

Phase 2  – Personalised FL with dual branches
---------------------------------------------
Every client is upgraded to **DigitModel_DB** by *duplicating* the weights
learned in Phase 1:

  • convolutional layers  →  copied into *both* ``shared_conv`` and
    ``local_conv`` branches;
  • 2048-D projection head  →  copied into *both* ``shared_proj`` and
    ``local_proj`` branches;
  • classifier head  →  copied into ``final_classifier``.

During federated training only the **shared** branch (``shared_conv`` &
``shared_proj``) is aggregated across clients. All other parameters – the
local branch and classifier – remain private, allowing each client to
retain its personalised expertise while still benefiting from global
knowledge.
"""

import os, sys, copy, time, argparse
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# repo root so that "import nets, utils" works no matter where the script lives
# -----------------------------------------------------------------------------
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from nets.models import DigitModel, DigitModel_DB  # type: ignore
from utils import data_utils  # type: ignore

# -----------------------------------------------------------------------------
# Data loading helpers (identical to fed_digits / single_digits)
# -----------------------------------------------------------------------------

def prepare_data(percent: float, batch_size: int):
    """Return *aligned* train / test loaders for the five Digits datasets."""
    # unified normalisation for all domains
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform_mnist = transforms.Compose([
        transforms.Grayscale(3), transforms.ToTensor(), norm
    ])
    transform_svhn = transforms.Compose([
        transforms.Resize((28, 28)), transforms.ToTensor(), norm
    ])
    transform_usps = transforms.Compose([
        transforms.Resize((28, 28)), transforms.Grayscale(3), transforms.ToTensor(), norm
    ])
    transform_synth = transforms.Compose([
        transforms.Resize((28, 28)), transforms.ToTensor(), norm
    ])
    transform_mnistm = transforms.Compose([
        transforms.ToTensor(), norm
    ])

    # datasets ---------------------------------------------------------------
    mnist_train = data_utils.DigitsDataset("../data/MNIST", 1, percent, None, True,  transform_mnist)
    mnist_test  = data_utils.DigitsDataset("../data/MNIST", 1, percent, None, False, transform_mnist)

    svhn_train  = data_utils.DigitsDataset("../data/SVHN", 3, percent, None, True,  transform_svhn)
    svhn_test   = data_utils.DigitsDataset("../data/SVHN", 3, percent, None, False, transform_svhn)

    usps_train  = data_utils.DigitsDataset("../data/USPS", 1, percent, None, True,  transform_usps)
    usps_test   = data_utils.DigitsDataset("../data/USPS", 1, percent, None, False, transform_usps)

    synth_train = data_utils.DigitsDataset("../data/SynthDigits", 3, percent, None, True,  transform_synth)
    synth_test  = data_utils.DigitsDataset("../data/SynthDigits", 3, percent, None, False, transform_synth)

    mnistm_train = data_utils.DigitsDataset("../data/MNIST_M", 3, percent, None, True,  transform_mnistm)
    mnistm_test  = data_utils.DigitsDataset("../data/MNIST_M", 3, percent, None, False, transform_mnistm)

    train_loaders = [
        torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=True)
        for d in [mnist_train, svhn_train, usps_train, synth_train, mnistm_train]
    ]
    test_loaders = [
        torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False)
        for d in [mnist_test, svhn_test, usps_test, synth_test, mnistm_test]
    ]

    return train_loaders, test_loaders

# -----------------------------------------------------------------------------
# Training / evaluation utilities
# -----------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, loss_fn, device):
    """Train **one** epoch; return (loss, acc)."""
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        optimizer.zero_grad()
        x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return epoch_loss / len(loader), correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
            logits = model(x)
            loss += loss_fn(logits, y).item()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return loss / len(loader), correct / total

# -----------------------------------------------------------------------------
# Weight transfer DigitModel  →  DigitModel_DB
# -----------------------------------------------------------------------------

def copy_single_to_dual(src: DigitModel, dst: DigitModel_DB):
    """Populate *dst* with weights from *src* (duplicate into both branches)."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()

    # ---- 1) convolutional stack ------------------------------------------------
    for name in [
        "conv1", "bn1", "conv2", "bn2", "conv3", "bn3"
    ]:
        for suffix in [".weight", ".bias", ".running_mean", ".running_var", ".num_batches_tracked"]:
            key = name + suffix
            if key in src_sd:  # BN layers include running_* keys; Conv layers do not
                for branch in ["shared_conv", "local_conv"]:
                    dst_key = f"{branch}.{key}"
                    dst_sd[dst_key] = src_sd[key].clone()

    # ---- 2) 2048-D projection head -------------------------------------------
    for name in ["fc1", "bn4"]:
        for suffix in [".weight", ".bias", ".running_mean", ".running_var", ".num_batches_tracked"]:
            key = name + suffix
            if key in src_sd:
                for branch in ["shared_proj", "local_proj"]:
                    dst_key = f"{branch}.{key}"
                    dst_sd[dst_key] = src_sd[key].clone()

    # ---- 3) final classifier ---------------------------------------------------
    for key in [
        "fc2.weight", "fc2.bias",
        "bn5.weight", "bn5.bias", "bn5.running_mean", "bn5.running_var", "bn5.num_batches_tracked",
        "fc3.weight", "fc3.bias",
    ]:
        if key in src_sd:
            dst_sd[f"final_classifier.{key}"] = src_sd[key].clone()

    dst.load_state_dict(dst_sd)

# -----------------------------------------------------------------------------
# Federated *shared-branch* aggregation
# -----------------------------------------------------------------------------

def aggregate_shared(server, clients: List[DigitModel_DB], weights):
    """FedAvg over *shared_conv/* and *shared_proj/*, *excluding* all BatchNorm params."""
    with torch.no_grad():
        for k, tensor in server.state_dict().items():
            if (not k.startswith(("shared_conv", "shared_proj"))) or ("bn" in k) or ("running_" in k) or ("num_batches_tracked" in k):
                # skip private branch & classifier
                continue

            # --- aggregate floating tensors (weights / biases) ---------------
            if tensor.dtype.is_floating_point:
                avg = sum(w * c.state_dict()[k].float() for w, c in zip(weights, clients))
                tensor.copy_(avg)
            else:  # integer tensors (very rare in shared_conv/proj)
                tensor.copy_(clients[0].state_dict()[k])

            # broadcast back
            for c in clients:
                c.state_dict()[k].copy_(tensor)
    return server, clients

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    # Phase-1 hyper-parameters ---------------------------------------------------
    parser.add_argument("--pretrain_epochs", type=int, default=150,
                        help="local warm-up epochs per client (DigitModel)")
    parser.add_argument("--pretrain_lr", type=float, default=1e-2)

    # Phase-2 (federated) hyper-parameters --------------------------------------
    parser.add_argument("--fl_iters", type=int, default=150,
                        help="number of communication rounds")
    parser.add_argument("--wk_iters", type=int, default=1,
                        help="local epochs per round (DigitModel_DB)")
    parser.add_argument("--fl_lr", type=float, default=1e-2)

    # misc ----------------------------------------------------------------------
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--percent", type=float, default=0.1)
    parser.add_argument("--save_path", type=str, default="../checkpoint/digits/twophase_digits")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # reproducibility -----------------------------------------------------------
    torch.manual_seed(0); np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    # data ----------------------------------------------------------------------
    train_loaders, test_loaders = prepare_data(args.percent, args.batch)
    domains = ["MNIST", "SVHN", "USPS", "Synth", "MNIST-M"]
    n_clients = len(domains)
    client_weights = [1.0 / n_clients] * n_clients

    loss_fn = nn.CrossEntropyLoss()

    # ======================================================================
    # PHASE 1 – LOCAL WARM-UP (DigitModel)
    # ======================================================================
    print("\n===== Phase 1: local warm-up (DigitModel) =====")
    single_models = []
    for idx, (train_loader, test_loader, domain) in enumerate(zip(train_loaders, test_loaders, domains)):
        print(f"\n[Client-{idx}] {domain}")
        model = DigitModel().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.pretrain_lr)
        for epoch in range(args.pretrain_epochs):
            loss, acc = run_epoch(model, train_loader, optimizer, loss_fn, device)
            if (epoch + 1) % 25 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{args.pretrain_epochs}  »  loss {loss:.4f}  •  acc {acc:.4f}")
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"  →  warm-up test accuracy: {test_acc:.4f}")
        single_models.append(model)

    # ======================================================================
    # PHASE 2 – FEDERATED TRAINING (DigitModel_DB)
    # ======================================================================
    print("\n===== Phase 2: personalised federated training (DigitModel_DB) =====")

    # (1)  build DigitModel_DB instances and copy weights -------------------
    client_models = []
    for sm in single_models:
        db = DigitModel_DB().to(device)
        copy_single_to_dual(sm, db)
        client_models.append(db)
        del sm  # free memory
    torch.cuda.empty_cache()

    # (2)  federated optimisation loop --------------------------------------
    server_model = copy.deepcopy(client_models[0]).to(device)

    for comm_round in range(args.fl_iters):
        print(f"\n----- Communication round {comm_round+1}/{args.fl_iters} -----")
        # local updates ------------------------------------------------------
        for client_idx, (model, train_loader) in enumerate(zip(client_models, train_loaders)):
            optimizer = optim.Adam(model.parameters(), lr=args.fl_lr)
            for _ in range(args.wk_iters):
                loss, acc = run_epoch(model, train_loader, optimizer, loss_fn, device)
            print(f"Client {client_idx} | train-acc {acc:.4f}")

        # aggregation --------------------------------------------------------
        server_model, client_models = aggregate_shared(server_model, client_models, client_weights)

        # evaluation ---------------------------------------------------------
        for idx, (model, test_loader, dom) in enumerate(zip(client_models, test_loaders, domains)):
            test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
            print(f" {dom:<8s} | test-acc {test_acc:.4f}")

    # ======================================================================
    # SAVE FINAL CHECKPOINTS -------------------------------------------------
    # ======================================================================
    os.makedirs(args.save_path, exist_ok=True)
    ckpt_path = os.path.join(args.save_path, "twophase_final.pt")
    torch.save({
        "server_model": server_model.state_dict(),
        **{f"client_{i}": m.state_dict() for i, m in enumerate(client_models)},
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
