# FGSP: Flatness-Guided Structured Pruning

Post-training structured pruning method that uses SAM-style perturbations as a
sensitivity probe. Prunes neuron clusters whose removal causes only a small loss
increase (i.e. clusters in flat, redundant regions), inducing robustness without
adversarial training.

**Paper:** *Sparsify to Robustify: Flatness-Guided Structured Pruning for Robust Neural Networks*

## Setup

```bash
cd ~
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running `prune.py`

### All available flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--arch` | `resnet18` | `resnet18`, `wrn28_10`, `resnet56`, `preact_resnet18` |
| `--checkpoint` | *(auto)* | Path to pretrained `.pth`. Auto-resolved from `models/` if not set |
| `--models_dir` | `models` | Directory containing pretrained checkpoints |
| `--prune_fraction` | `0.7` | Target sparsity. `0.7` = 70%, `0.9` = 90% |
--rho` | `0.3` | SAM perturbation radius ρ (paper Table 8: `0.3`) |
| `--num_clusters` | `20` | Agglomerative clusters per layer (paper: `20`) |
| `--fine_tune_epochs` | `3` | Fine-tuning epochs after each layer is pruned |
| `--fine_tune_lr` | `0.01` | Fine-tuning learning rate |
| `--batch_size` | `128` | DataLoader batch size |
| `--device` | *(auto)* | e.g. `cuda:0`, `cuda:1`, `cpu`. Auto-detected if unset |
| `--seed` | `42` | Random seed for reproducibility |
| `--output_dir` | `outputs` | Root directory for logs and saved models |
| `--tag` | *(empty)* | Optional string appended to output filenames |
| `--skip_sensitivity` | `False` | Skip Phase 1 and load existing sensitivity CSV from disk |

---

## Exact Commands to Replicate Paper Results

### CIFAR-10 — ResNet-18

> **Note:** The notebook uses `rho=2` and `fine_tune_epochs=40`.
> The paper (Table 8, Appendix A.2) reports `rho=0.3` and `fine_tune_epochs=3`.
> Use the commands below to match each exactly.

**To match notebook V3 exactly (what produced your saved models):**
```bash
# 10% sparsity
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.1 --rho 2 --fine_tune_epochs 40 --device cuda:1

# 50% sparsity
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.5 --rho 2 --fine_tune_epochs 40 --device cuda:1

# 70% sparsity
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 --rho 2 --fine_tune_epochs 40 --device cuda:1

# 90% sparsity
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.9 --rho 2 --fine_tune_epochs 40 --device cuda:1
```

**To match paper hyperparameters (Table 8):**
```bash
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 --rho 0.3 --fine_tune_epochs 3 --device cuda:1
```

**Sweep all sparsity levels (shell loop):**
```bash
for pf in 0.1 0.5 0.7 0.9; do
    python prune.py --dataset cifar10 --arch resnet18 --prune_fraction $pf \--rho 2 --fine_tune_epochs 40 --device cuda:1
done
```

---

### CIFAR-10 — WRN-28-10 (main paper results, Tables 1–3)

```bash
for pf in 0.1 0.5 0.7; do
    python prune.py --dataset cifar10 --arch wrn28_10 --prune_fraction $pf \--rho 0.3 --fine_tune_epochs 3 --device cuda:1
done
```

---

### CIFAR-100 — WRN-28-10 (Tables 4–6)

> **Note:** The CIFAR-100 notebook uses `batch_size=256`, `rho=2`, `fine_tune_epochs=3`.

**To match notebook V3 CIFAR-100 exactly:**
```bash
# 10% sparsity
python prune.py --dataset cifar100 --arch wrn28_10 --prune_fraction 0.1 --rho 2 --fine_tune_epochs 3 --batch_size 256 --device cuda:1

# 50% sparsity
python prune.py --dataset cifar100 --arch wrn28_10 --prune_fraction 0.5 --rho 2 --fine_tune_epochs 3 --batch_size 256 --device cuda:1

# 70% sparsity
python prune.py --dataset cifar100 --arch wrn28_10 --prune_fraction 0.7 --rho 2 --fine_tune_epochs 3 --batch_size 256 --device cuda:1
```

---

## Resuming / Skipping Phase 1

Phase 1 (sensitivity analysis) is the expensive part — it scores every cluster
in every layer. Once it runs once, the CSV is saved automatically. If you want
to re-run pruning with different fine-tuning settings without redoing Phase 1:

```bash
# First run — computes and saves sensitivity CSV
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 --rho 2 --fine_tune_epochs 40 --device cuda:1

# Re-run Phase 2 only (loads existing CSV, skips Phase 1)
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 --rho 2 --fine_tune_epochs 40 --device cuda:1 --skip_sensitivity
```

The sensitivity CSV is named by run parameters, s--rho` and `--num_clusters`
must match the original run when using `--skip_sensitivity`.

## Output Files

For a run with `--arch resnet18 --dataset cifar10 --prune_fraction 0--rho 2 --fine_tune_epochs 40`:

| File | Description |
|------|-------------|
| `outputs/logs/fgsp_resnet18_cifar10_rho2_K20_ft40e_07PR_sensitivity.csv` | Phase 1 cluster scores. Cached — reused on `--skip_sensitivity` |
| `outputs/models/fgsp_resnet18_cifar10_rho2_K20_ft40e_07PR.pth` | Final clean pruned checkpoint (masks baked in, no `_orig`/`_mask` keys) |

The sensitivity CSV has columns: `layer_name`, `cluster_id`, `sensitivity_score`,
`neuron_indices`, `num_neurons`.

---

## Default Checkpoint Lookup

If `--checkpoint` is not passed, `prune.py` looks for:

| arch + dataset | Expected filename in `models/` |
|---|---|
| `resnet18` + `cifar10` | `resnet18_c10_no_sam_100e.pth` |
| `resnet18` + `cifar100` | `resnet18_c100_no_sam_100e.pth` |
| `wrn28_10` + `cifar10` | `wide-resnet_c10_no_sam_40e.pth` |
| `wrn28_10` + `cifar100` | `wide-resnet_c100_no_sam_40e.pth` |
| `resnet56` + `cifar10` | `resnet56_c10_no_sam_100e.pth` |
| `preact_resnet18` + `cifar10` | `preact_resnet18_c10_no_sam_100e.pth` |

To use a different checkpoint name, pass `--checkpoint models/your_model.pth`.