#!/usr/bin/env python3
"""Verify Digits two-phase checkpoint.

Loads a saved checkpoint (e.g. checkpoint/digits/twophase_digits/twophase_final.pt)
and reproduces the per-domain test accuracies printed by federated/fed_digits_twophase.py.

Expected checkpoint format (as saved by fed_digits_twophase.py):
  {
    "server_model": OrderedDict(...),
    "client_0": OrderedDict(...),
    ...
    "client_4": OrderedDict(...),
  }

Run:
  uv run python scripts/verify_digits_twophase.py \
    --ckpt checkpoint/digits/twophase_digits/twophase_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# repo root so that "import nets, utils" works no matter where the script lives
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from nets.models import DigitModel_DB  # type: ignore
from utils import data_utils  # type: ignore


DOMAINS: List[str] = ["MNIST", "SVHN", "USPS", "Synth", "MNIST-M"]


def _load_checkpoint(path: str) -> Dict[str, object]:
    # Prefer weights_only=True for safety (Torch >= 2.0).
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def prepare_test_loaders(data_root: str, batch_size: int) -> List[torch.utils.data.DataLoader]:
    """Create the same test loaders as fed_digits_twophase.prepare_data()."""
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform_mnist = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        norm,
    ])
    transform_svhn = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        norm,
    ])
    transform_usps = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        norm,
    ])
    transform_synth = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        norm,
    ])
    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    # NOTE: percent is irrelevant for test split (DigitsDataset ignores it when train=False)
    percent = 0.1

    mnist_test = data_utils.DigitsDataset(os.path.join(data_root, "MNIST"), 1, percent, None, False, transform_mnist)
    svhn_test = data_utils.DigitsDataset(os.path.join(data_root, "SVHN"), 3, percent, None, False, transform_svhn)
    usps_test = data_utils.DigitsDataset(os.path.join(data_root, "USPS"), 1, percent, None, False, transform_usps)
    synth_test = data_utils.DigitsDataset(os.path.join(data_root, "SynthDigits"), 3, percent, None, False, transform_synth)
    mnistm_test = data_utils.DigitsDataset(os.path.join(data_root, "MNIST_M"), 3, percent, None, False, transform_mnistm)

    datasets = [mnist_test, svhn_test, usps_test, synth_test, mnistm_test]
    return [
        torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=False)
        for d in datasets
    ]


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)
        logits = model(x)
        loss += loss_fn(logits, y).item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return loss / max(1, len(loader)), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join("checkpoint", "digits", "twophase_digits", "twophase_final.pt"),
        help="Path to twophase_final.pt",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=os.path.join(BASE_PATH, "data"),
        help="Path to data/ directory containing MNIST, SVHN, USPS, SynthDigits, MNIST_M",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument("--eval-server", action="store_true", help="Also evaluate server_model")
    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(BASE_PATH, ckpt_path)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    ckpt = _load_checkpoint(ckpt_path)

    # Prepare loaders
    test_loaders = prepare_test_loaders(args.data_root, args.batch)

    loss_fn = nn.CrossEntropyLoss()

    def _eval_state_dict(sd_key: str, sd) -> None:
        model = DigitModel_DB().to(device)
        model.load_state_dict(sd)
        for dom, loader in zip(DOMAINS, test_loaders):
            _loss, acc = evaluate(model, loader, loss_fn, device)
            print(f" {dom:<8s} | test-acc {acc:.4f}")

    # Evaluate client models (matches training output)
    for i in range(len(DOMAINS)):
        key = f"client_{i}"
        if key not in ckpt:
            raise KeyError(f"Missing key in checkpoint: {key}. Available keys: {list(ckpt.keys())}")

    for i, dom in enumerate(DOMAINS):
        key = f"client_{i}"
        model = DigitModel_DB().to(device)
        model.load_state_dict(ckpt[key])
        _loss, acc = evaluate(model, test_loaders[i], loss_fn, device)
        print(f" {dom:<8s} | test-acc {acc:.4f}")

    if args.eval_server:
        if "server_model" not in ckpt:
            raise KeyError("Missing key in checkpoint: server_model")
        print("\n[server_model] evaluated on each domain test set:")
        _eval_state_dict("server_model", ckpt["server_model"])


if __name__ == "__main__":
    main()
