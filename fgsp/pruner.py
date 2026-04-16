"""

Phase 2: Structured Pruning and Fine-tuning.

Clusters are ranked by sensitivity score (ascending). The least sensitive
clusters (those in flat regions of the loss surface) are removed via
structured channel pruning. After each layer is pruned, a short fine-tuning
pass recovers accuracy. Finally, pruning masks are baked into the weights
to produce a clean, sparse checkpoint.
"""

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as torch_prune


# Layers excluded from pruning (must match analyzer.py)
SKIP_LAYERS = {"conv1", "linear", "conv", "fc"}


# ===========================================================================
# Fine-tuning
# ===========================================================================

def fine_tune(model: nn.Module, train_loader, criterion, device: str,
              normalize_fn, epochs: int, lr: float = 0.01) -> nn.Module:
    """
    Short SGD fine-tuning with cosine LR decay.
    Called after each layer is pruned to stabilize the network.
    """
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(normalize_fn(x)), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"    [fine-tune {epoch+1}/{epochs}] "
              f"loss={total_loss / len(train_loader):.4f}")
    model.eval()
    return model


# ===========================================================================
# Mask helpers
# ===========================================================================

def _parse_indices(raw) -> list:
    """
    Safely parse neuron_indices that may have been round-tripped through CSV.
    Handles Python lists, strings like '[1, 2, 3]', and numpy int64 repr.
    """
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return [int(x) for x in re.findall(r'\d+', str(raw))]


def _apply_channel_mask(module: nn.Module, pruned_indices: list):
    """
    Zeros out the selected output channels (neurons) in a Conv2d or Linear
    layer using PyTorch's custom_from_mask utility.
    Also masks the corresponding bias entries if present.
    """
    # Remove any existing pruning hooks first (idempotent)
    for attr in ('weight', 'bias'):
        if hasattr(module, f'{attr}_mask'):
            torch_prune.remove(module, attr)

    # Weight mask: zero entire output channel for each pruned index
    w_mask = torch.ones_like(module.weight)
    for idx in pruned_indices:
        if idx < w_mask.shape[0]:
            w_mask[idx] = 0          # works for both Conv2d and Linear
    torch_prune.custom_from_mask(module, 'weight', w_mask)

    # Bias mask
    if module.bias is not None:
        b_mask = torch.ones_like(module.bias)
        for idx in pruned_indices:
            if idx < b_mask.shape[0]:
                b_mask[idx] = 0
        torch_prune.custom_from_mask(module, 'bias', b_mask)


# ===========================================================================
# Phase 2: Iterative layer-wise pruning
# ===========================================================================

def iterative_pruning(
    model: nn.Module,
    sensitivity_df,
    train_loader,
    criterion,
    device: str,
    normalize_fn,
    prune_fraction: float,
    fine_tune_epochs: int = 3,
    fine_tune_lr: float = 0.01,
) -> nn.Module:
    """
    Layer-wise structured pruning guided by SAM sensitivity scores.

    For each layer:
      1. Sort clusters by sensitivity_score (ascending = least sensitive first).
      2. Prune the N least sensitive clusters (N ∝ prune_fraction).
      3. Fine-tune the network briefly to recover accuracy.

    Args:
        model:            Pretrained dense model.
        sensitivity_df:   DataFrame from NeuronAnalyzer.analyze().
        train_loader:     Training DataLoader for fine-tuning.
        criterion:        Loss function (CrossEntropyLoss).
        device:           'cuda:X' or 'cpu'.
        normalize_fn:     Dataset normalization callable.
        prune_fraction:   Target sparsity, e.g. 0.7 = 70%.
        fine_tune_epochs: Epochs of fine-tuning per layer.
        fine_tune_lr:     Learning rate for fine-tuning.

    Returns:
        Pruned (and fine-tuned) model with torch pruning hooks applied.
        Call bake_masks() on the saved state_dict for a clean checkpoint.
    """
    for layer_name, group in sensitivity_df.groupby('layer_name'):
        if layer_name in SKIP_LAYERS:
            print(f"  [skip] {layer_name}")
            continue

        # ---- Parse cluster info ----
        cluster_info = []
        total_neurons = 0
        for _, row in group.iterrows():
            indices = _parse_indices(row['neuron_indices'])
            total_neurons += len(indices)
            cluster_info.append({
                'sensitivity_score': float(row['sensitivity_score']),
                'neuron_indices': indices,
                'num_neurons': len(indices),
            })

        # Each of the 20 clusters represents ~5% of the layer.
        # prune_fraction → number of clusters to remove.
        num_clusters = len(cluster_info)
        num_to_prune = min(
            int(np.ceil(prune_fraction / 0.05)),
            num_clusters
        )

        # Sort ascending: least sensitive first
        cluster_info.sort(key=lambda c: c['sensitivity_score'])

        # Collect indices of neurons to prune
        pruned_indices = []
        for c in cluster_info[:num_to_prune]:
            pruned_indices.extend(c['neuron_indices'])

        pct = 100.0 * len(pruned_indices) / total_neurons
        print(f"\n  Layer {layer_name}: pruning {len(pruned_indices)}/{total_neurons} "
              f"neurons ({pct:.1f}%) from {num_to_prune}/{num_clusters} clusters")

        # ---- Apply mask ----
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, (nn.Conv2d, nn.Linear)):
                _apply_channel_mask(module, pruned_indices)
                n_zero = (module.weight.data == 0).sum().item()
                print(f"    → {n_zero} weights zeroed in '{name}'")
                break

        # ---- Fine-tune after this layer ----
        print(f"  Fine-tuning ({fine_tune_epochs} epochs)...")
        model = fine_tune(
            model, train_loader, criterion, device, normalize_fn,
            fine_tune_epochs, fine_tune_lr
        )

    return model


# ===========================================================================
# Post-pruning utilities
# ===========================================================================

def bake_masks(state_dict: dict) -> dict:
    """
    Folds PyTorch pruning hooks (weight_orig * weight_mask) into clean
    weight tensors. The returned dict can be loaded into a standard model
    with no pruning hooks attached.
    """
    clean = {}
    for k, v in state_dict.items():
        if k.endswith('_orig'):
            base = k[:-5]
            mask_key = base + '_mask'
            clean[base] = v * state_dict[mask_key] if mask_key in state_dict else v
        elif not k.endswith('_mask'):
            clean[k] = v
    return clean


def measure_sparsity(model: nn.Module) -> float:
    """Reports and returns the global weight sparsity of the model."""
    total = zeros = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight'):
            w = m.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    sparsity = 100.0 * zeros / total if total > 0 else 0.0
    print(f"  Sparsity: {zeros:,}/{total:,} weights zero ({sparsity:.2f}%)")
    return sparsity