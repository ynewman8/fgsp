"""
fgsp/analyzer.py

Phase 1: Clustered SAM Sensitivity Analysis.

For each prunable layer, neurons are grouped into K clusters via agglomerative
clustering on weight-vector geometry (Ward linkage on L2 norms). Each cluster
is scored by the increase in validation loss after a localized SAM-style
perturbation is applied to only that cluster's weights.

Clusters with low sensitivity scores are structurally redundant and will be
pruned in Phase 2.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering


# Layers excluded from pruning (input conv + output linear)
SKIP_LAYERS = {"conv1", "linear", "conv", "fc"}


class NeuronAnalyzer:
    """
    Computes flatness-based sensitivity scores for neuron clusters in each layer.

    Usage:
        analyzer = NeuronAnalyzer(model, criterion, device, normalize_fn,
                                  num_clusters=20, rho=0.3)
        sensitivity_df = analyzer.analyze(val_loader)
        sensitivity_df.to_csv("logs/sensitivity.csv", index=False)
    """

    def __init__(self, model: nn.Module, criterion, device: str,
                 normalize_fn, num_clusters: int = 20, rho: float = 0.3):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.normalize_fn = normalize_fn
        self.num_clusters = num_clusters
        self.rho = rho
        self._neuron_info = defaultdict(dict)

    # ------------------------------------------------------------------
    # Geometry-aware clustering (Ward linkage on weight vectors)
    # ------------------------------------------------------------------

    def _build_clusters(self, module: nn.Module, num_neurons: int):
        """
        Groups num_neurons into K balanced clusters using agglomerative
        clustering (Ward linkage) on flattened weight vectors.

        Returns a list of K lists, each containing neuron indices.
        """
        K = min(self.num_clusters, num_neurons)
        if K < 2:
            return None

        # Reshape each neuron's weights into a 1D vector for clustering
        weight_vectors = (
            module.weight.detach().cpu()
            .reshape(num_neurons, -1)
            .numpy()
        )

        clustering = AgglomerativeClustering(n_clusters=K, linkage='ward')
        raw_labels = clustering.fit_predict(weight_vectors)

        # Build balanced clusters (each cluster ≈ num_neurons / K)
        cluster_size = num_neurons // K
        remainder = num_neurons % K
        target_sizes = [
            cluster_size + (1 if i < remainder else 0) for i in range(K)
        ]

        clusters = [[] for _ in range(K)]
        sorted_indices = np.argsort(raw_labels)
        ptr = 0
        for i in range(K):
            while len(clusters[i]) < target_sizes[i] and ptr < num_neurons:
                clusters[i].append(int(sorted_indices[ptr]))
                ptr += 1

        # Redistribute any leftovers (edge cases)
        for i in range(K):
            while len(clusters[i]) < target_sizes[i]:
                for j in range(K):
                    if len(clusters[j]) > target_sizes[j]:
                        clusters[i].append(clusters[j].pop())
                        break

        return clusters

    # ------------------------------------------------------------------
    # SAM perturbation for a single cluster
    # ------------------------------------------------------------------

    def _perturb_cluster(self, module: nn.Module, neuron_indices: list):
        """
        Applies a SAM-style perturbation in-place to each neuron in the cluster:
            w_j ← w_j + ρ · ∇w_j L / ‖∇w_j L‖₂
        Gradient must already be computed before calling this.
        """
        for idx in neuron_indices:
            if module.weight.grad is None:
                continue
            g = module.weight.grad[idx]
            g_norm = torch.norm(g)
            if g_norm > 0:
                module.weight.data[idx] += self.rho * g / g_norm

    # ------------------------------------------------------------------
    # Main analysis loop
    # ------------------------------------------------------------------

    def analyze(self, dataloader) -> pd.DataFrame:
        """
        Runs Phase 1: computes sensitivity scores for all prunable layers.

        For each layer → for each cluster → perturb weights → measure loss
        increase → record as sensitivity score.

        Returns a DataFrame with columns:
            layer_name, cluster_id, sensitivity_score, neuron_indices, num_neurons
        """
        results = []
        self.model.eval()

        # Collect prunable layers
        layers = [
            (name, module)
            for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
            and name not in SKIP_LAYERS
        ]
        skipped = [
            name for name, module in self.model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
            and name in SKIP_LAYERS
        ]
        if skipped:
            print(f"  Skipping (input/output layers): {skipped}")

        for layer_name, module in layers:
            num_neurons = (
                module.out_channels if isinstance(module, nn.Conv2d)
                else module.out_features
            )

            clusters = self._build_clusters(module, num_neurons)
            if clusters is None:
                print(f"  [skip] {layer_name}: only {num_neurons} neurons, "
                      f"need ≥ 2 for clustering")
                continue

            K = len(clusters)
            sizes = Counter(len(c) for c in clusters)
            print(f"\n  Layer {layer_name}: {num_neurons} neurons → "
                  f"{K} clusters (~{num_neurons/K:.1f} per cluster) "
                  f"sizes={dict(sizes)}")

            for cluster_id, neuron_indices in enumerate(clusters):
                if not neuron_indices:
                    continue

                baseline_losses = []
                perturbed_losses = []
                original_state = deepcopy(module.state_dict())

                for x, y in dataloader:
                    x, y = x.to(self.device), y.to(self.device)

                    # ---- Baseline forward + backward ----
                    self.model.zero_grad()
                    out = self.model(self.normalize_fn(x))
                    loss = self.criterion(out, y)
                    loss.backward()

                    baseline_losses.extend(
                        F.cross_entropy(out, y, reduction='none')
                        .detach().cpu().numpy()
                    )

                    # ---- Apply SAM perturbation to this cluster ----
                    self._perturb_cluster(module, neuron_indices)

                    # ---- Perturbed forward (no grad needed) ----
                    with torch.no_grad():
                        p_out = self.model(self.normalize_fn(x))
                        perturbed_losses.extend(
                            F.cross_entropy(p_out, y, reduction='none')
                            .cpu().numpy()
                        )

                    # ---- Restore original weights ----
                    module.load_state_dict(original_state)

                # Sensitivity score: mean relative loss increase
                bl = np.array(baseline_losses)
                pl = np.array(perturbed_losses)
                sensitivity = np.mean(np.abs(pl - bl) / (bl + 1e-10))

                results.append({
                    'layer_name': layer_name,
                    'cluster_id': f"cluster_{cluster_id}",
                    'sensitivity_score': sensitivity,
                    'neuron_indices': neuron_indices,
                    'num_neurons': len(neuron_indices),
                })

        df = pd.DataFrame(results)
        print(f"\n  Sensitivity analysis complete: {len(df)} clusters scored "
              f"across {df['layer_name'].nunique()} layers.")
        return df