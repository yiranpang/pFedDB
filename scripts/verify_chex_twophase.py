#!/usr/bin/env python3
"""Verify OVCR checkpoint (PLENet local-branch zero).

This script evaluates a saved checkpoint on the OVCR test sets and prints per-client
Accuracy/AUC + averages.

Default checkpoint path:
    checkpoint/chex/twophase_chex/twophase_chex_final

Expected checkpoint format:
  {
    "model_0": state_dict,
    "model_1": state_dict,
    "model_2": state_dict,
    "model_3": state_dict,
    ...
  }
(or alternatively a server checkpoint with key "server_model").

Run:
    uv run python scripts/verify_chex_twophase.py \
        --ckpt checkpoint/chex/twophase_chex/twophase_chex_final
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from nets.models import PLENet_DenseNet_ShareCNN, PLENetReluBeforePoolWrapper  # type: ignore


DATASET_NAMES: List[str] = ["OVCR_chex", "OVCR_openi", "OVCR_rsna", "OVCR_vinbigdata"]


def _load_checkpoint(path: str) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        return torch.load(path, map_location="cpu")


def _default_data_root() -> str:
    # Workspace default
    return "/home/ypang1/expanse/pFedDB/data"


def prepare_test_loaders(data_root: str, batch_size: int) -> List[DataLoader]:
    transform_test = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    loaders: List[DataLoader] = []
    for name in DATASET_NAMES:
        test_dir = os.path.join(data_root, name, "test")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(
                f"OVCR test directory not found: {test_dir}\n"
                f"Set --data-root to the OVCR root containing {DATASET_NAMES}."
            )
        ds = datasets.ImageFolder(root=test_dir, transform=transform_test)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=False))
    return loaders


@torch.no_grad()
def test_with_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    model.eval()

    all_labels: List[int] = []
    all_preds: List[int] = []
    all_probs: List[List[float]] = []

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        all_labels.extend(y.detach().cpu().numpy().tolist())
        all_preds.extend(pred.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    labels = np.asarray(all_labels)
    preds = np.asarray(all_preds)
    probs_np = np.asarray(all_probs)

    acc = float(accuracy_score(labels, preds))

    try:
        if probs_np.ndim == 2 and probs_np.shape[1] == 2:
            auc = float(roc_auc_score(labels, probs_np[:, 1]))
        else:
            auc = float(roc_auc_score(labels, probs_np, multi_class="ovr"))
    except Exception:
        auc = -1.0

    return acc, auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=os.path.join("checkpoint", "chex", "twophase_chex", "twophase_chex_final"),
        help="Checkpoint path (default: checkpoint/chex/twophase_chex/twophase_chex_final)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=_default_data_root(),
        help="OVCR dataset root containing OVCR_chex/OVCR_openi/OVCR_rsna/OVCR_vinbigdata",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto)")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limit evaluation to N batches per dataset (0 = full test set). Useful for smoke tests.",
    )
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

    test_loaders = prepare_test_loaders(args.data_root, args.batch)

    ckpt = _load_checkpoint(ckpt_path)

    client_num = len(DATASET_NAMES)
    has_client_models = any(f"model_{i}" in ckpt for i in range(client_num))

    print(f"Using device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Checkpoint: {ckpt_path}")

    per_client_acc: List[float] = []
    per_client_auc: List[float] = []

    if has_client_models:
        for i, name in enumerate(DATASET_NAMES):
            key = f"model_{i}"
            if key not in ckpt:
                raise KeyError(f"Missing key in checkpoint: {key}. Available keys: {list(ckpt.keys())}")

            model = PLENet_DenseNet_ShareCNN()
            model.load_state_dict(ckpt[key])
            model = model.to(device)

            eval_model: torch.nn.Module = PLENetReluBeforePoolWrapper(model)
            acc, auc = test_with_metrics(eval_model, test_loaders[i], device, max_batches=args.max_batches)

            per_client_acc.append(acc)
            per_client_auc.append(auc)
            print(f"{name:<18s} Acc={acc:.4f} AUC={auc:.4f}")
    else:
        if "server_model" not in ckpt:
            raise KeyError(
                "Checkpoint does not contain client models (model_i) or server_model. "
                f"Keys: {list(ckpt.keys())}"
            )

        model = PLENet_DenseNet_ShareCNN()
        model.load_state_dict(ckpt["server_model"])  # type: ignore[arg-type]
        model = model.to(device)
        eval_model: torch.nn.Module = PLENetReluBeforePoolWrapper(model)

        for i, name in enumerate(DATASET_NAMES):
            acc, auc = test_with_metrics(eval_model, test_loaders[i], device, max_batches=args.max_batches)
            per_client_acc.append(acc)
            per_client_auc.append(auc)
            print(f"{name:<18s} Acc={acc:.4f} AUC={auc:.4f}")

    avg_acc = float(np.mean(per_client_acc))
    valid_aucs = [a for a in per_client_auc if a > 0]
    avg_auc = float(np.mean(valid_aucs)) if valid_aucs else -1.0

    print(f"AVG Acc={avg_acc:.4f} | AVG AUC={avg_auc:.4f}")


if __name__ == "__main__":
    main()
