"""
prune.py — FGSP: Flatness-Guided Structured Pruning
=====================================================
Main entry point. Run from your home directory (~/) on the server.

Usage examples
--------------
# CIFAR-10, ResNet-18, 70% sparsity (paper's main result)
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 --device cuda:1

# CIFAR-100, WRN-28-10, 50% sparsity
python prune.py --dataset cifar100 --arch wrn28_10 --prune_fraction 0.5 --device cuda:1

# Explicit checkpoint + 40-epoch fine-tuning (CIFAR-10 RN18 as in paper)
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.9 \
    --checkpoint models/resnet18_c10_no_sam_100e.pth \
    --fine_tune_epochs 40 --rho 0.3 --device cuda:1

# Sweep all sparsity levels
for pf in 0.1 0.5 0.7 0.9; do
    python prune.py --dataset cifar10 --arch resnet18 --prune_fraction $pf --device cuda:1
done

Supported architectures : resnet18, wrn28_10, resnet56, preact_resnet18
Supported datasets       : cifar10, cifar100
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the ICML repo importable (for load_dataset, normalize_cifar, etc.)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import load_dataset, normalize_cifar, normalize_cifar100  # noqa

from fgsp.analyzer import NeuronAnalyzer
from fgsp.pruner import iterative_pruning, bake_masks, measure_sparsity
from fgsp.models import build_model, default_checkpoint


def get_args():
    parser = argparse.ArgumentParser(
        description="FGSP: Flatness-Guided Structured Pruning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--arch', default='resnet18',
                        choices=['resnet18', 'wrn28_10', 'resnet56', 'preact_resnet18'])
    parser.add_argument('--checkpoint', default=None,
                        help='Path to pretrained .pth. Auto-resolved from models/ if unset.')
    parser.add_argument('--models_dir', default='models')
    parser.add_argument('--prune_fraction', type=float, default=0.7,
                        help='Target pruning fraction (0.7 = 70%% sparsity)')
    parser.add_argument('--rho', type=float, default=0.3,
                        help='SAM perturbation radius rho')
    parser.add_argument('--num_clusters', type=int, default=20)
    parser.add_argument('--fine_tune_epochs', type=int, default=3)
    parser.add_argument('--fine_tune_lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default=None,
                        help='e.g. cuda:0, cuda:1, cpu. Auto-detected if unset.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--tag', default='',
                        help='Optional string appended to output filenames')
    parser.add_argument('--skip_sensitivity', action='store_true',
                        help='Load existing sensitivity CSV instead of recomputing')
    return parser.parse_args()


def get_normalize_fn(dataset):
    return normalize_cifar if dataset == 'cifar10' else normalize_cifar100


def evaluate_clean(model, loader, device, normalize_fn):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(normalize_fn(x)).argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"  Clean accuracy: {acc:.4f}  ({100*acc:.2f}%)")
    return acc


def make_run_name(args):
    pr_int = int(args.prune_fraction * 10)
    tag = f"_{args.tag}" if args.tag else ""
    return (f"fgsp_{args.arch}_{args.dataset}"
            f"_rho{args.rho}_K{args.num_clusters}"
            f"_ft{args.fine_tune_epochs}e_{pr_int:02d}PR{tag}")


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device
    num_classes = 10 if args.dataset == 'cifar10' else 100
    normalize_fn = get_normalize_fn(args.dataset)
    run_name = make_run_name(args)

    log_dir = os.path.join(args.output_dir, 'logs')
    model_dir = os.path.join(args.output_dir, 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    sensitivity_csv = os.path.join(log_dir, f"{run_name}_sensitivity.csv")
    pruned_ckpt = os.path.join(model_dir, f"{run_name}.pth")

    print(f"\n{'='*65}")
    print(f"  FGSP | arch={args.arch} dataset={args.dataset} "
          f"sparsity={int(args.prune_fraction*100)}% device={device}")
    print(f"  rho={args.rho}  K={args.num_clusters}  "
          f"fine_tune_epochs={args.fine_tune_epochs}")
    print(f"{'='*65}\n")

    # Data
    train_loader, test_loader = load_dataset(args.dataset, args.batch_size)
    criterion = nn.CrossEntropyLoss()

    # Model
    model = build_model(args.arch, num_classes, device)
    ckpt_path = args.checkpoint or default_checkpoint(
        args.arch, args.dataset, args.models_dir)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: '{ckpt_path}'\n"
            f"Pass --checkpoint /path/to/model.pth or check --models_dir."
        )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[Model] Loaded: {ckpt_path}")

    print("\n[Baseline]")
    evaluate_clean(model, test_loader, device, normalize_fn)
    measure_sparsity(model)

    # Phase 1: SAM Sensitivity Analysis
    print("\n[Phase 1] SAM Sensitivity Analysis")
    if args.skip_sensitivity and os.path.isfile(sensitivity_csv):
        import pandas as pd
        print(f"  Loading: {sensitivity_csv}")
        sensitivity_df = pd.read_csv(sensitivity_csv)
    else:
        analyzer = NeuronAnalyzer(
            model, criterion, device, normalize_fn,
            num_clusters=args.num_clusters, rho=args.rho,
        )
        sensitivity_df = analyzer.analyze(test_loader)
        sensitivity_df.to_csv(sensitivity_csv, index=False)
        print(f"  Saved → {sensitivity_csv}")

    # Phase 2: Structured Pruning + Fine-tuning
    print("\n[Phase 2] Structured Pruning & Fine-tuning")
    pruned_model = iterative_pruning(
        model, sensitivity_df,
        train_loader, criterion, device, normalize_fn,
        prune_fraction=args.prune_fraction,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
    )

    # Save clean checkpoint
    clean_sd = bake_masks(pruned_model.state_dict())
    torch.save(clean_sd, pruned_ckpt)
    print(f"\n[Save] Pruned model → {pruned_ckpt}")

    # Final evaluation
    print("\n[Final Evaluation]")
    final_model = build_model(args.arch, num_classes, device)
    final_model.load_state_dict(clean_sd)
    evaluate_clean(final_model, test_loader, device, normalize_fn)
    measure_sparsity(final_model)

    print(f"\n{'='*65}")
    print(f"  Done. Sensitivity: {sensitivity_csv}")
    print(f"  Model:            {pruned_ckpt}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()