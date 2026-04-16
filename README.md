# FGSP: Flatness-Guided Structured Pruning

Post-training structured pruning that uses SAM-style perturbations as a sensitivity probe. Clusters of neurons whose removal causes only a small loss increase — i.e. those in flat, redundant regions of the loss landscape — are pruned, inducing robustness without adversarial training.

**Paper:** *Sparsify to Robustify: Flatness-Guided Structured Pruning for Robust Neural Networks* (ICML submission)

---

## Setup

```bash
git clone https://github.com/ynewman8/fgsp.git
cd fgsp
pip install -r requirements.txt
```

Data downloads automatically to `./data/` on first run.

Place pretrained checkpoints in a `models/` directory before running (see [Default Checkpoint Lookup](#default-checkpoint-lookup)).

---

## Usage

```bash
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 --device cuda:0
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10` or `cifar100` |
| `--arch` | `resnet18` | `resnet18`, `wrn28_10`, `resnet56`, `preact_resnet18` |
| `--checkpoint` | *(auto)* | Path to pretrained `.pth`. Auto-resolved from `models/` if not set |
| `--models_dir` | `models` | Directory containing pretrained checkpoints |
| `--prune_fraction` | `0.7` | Target sparsity — `0.7` = 70%, `0.9` = 90% |
| `--rho` | `2` | SAM perturbation radius ρ, applied independently per neuron |
| `--num_clusters` | `20` | Agglomerative clusters per layer |
| `--fine_tune_epochs` | `3` | Fine-tuning epochs after each layer is pruned |
| `--fine_tune_lr` | `0.01` | Fine-tuning learning rate |
| `--batch_size` | `128` | DataLoader batch size |
| `--device` | *(auto)* | e.g. `cuda:0`, `cuda:1`, `cpu` |
| `--seed` | `42` | Random seed |
| `--output_dir` | `outputs` | Root directory for logs and saved models |
| `--tag` | *(empty)* | Optional string appended to output filenames |
| `--skip_sensitivity` | `False` | Skip Phase 1 and load an existing sensitivity CSV |

---

## Replicating Paper Results

### CIFAR-10 — ResNet-18

```bash
for pf in 0.1 0.5 0.7 0.9; do
    python prune.py --dataset cifar10 --arch resnet18 --prune_fraction $pf \
        --rho 2 --fine_tune_epochs 40 --batch_size 128 --device cuda:0
done
```

### CIFAR-10 — WRN-28-10 (Tables 1–3)

```bash
for pf in 0.1 0.5 0.7; do
    python prune.py --dataset cifar10 --arch wrn28_10 --prune_fraction $pf \
        --rho 2 --fine_tune_epochs 40 --batch_size 128 --device cuda:0
done
```

### CIFAR-100 — WRN-28-10 (Tables 4–6)

```bash
for pf in 0.1 0.5 0.7; do
    python prune.py --dataset cifar100 --arch wrn28_10 --prune_fraction $pf \
        --rho 2 --fine_tune_epochs 3 --batch_size 128 --device cuda:0
done
```

---

## Skipping Phase 1 (Sensitivity Caching)

Phase 1 (SAM sensitivity scoring) is the expensive step. Once it runs, the scores are saved as a CSV and can be reused — useful when iterating on fine-tuning settings without redoing the full analysis.

```bash
# First run — scores all clusters and saves CSV
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 \
    --rho 2 --fine_tune_epochs 40 --device cuda:0

# Subsequent runs — skip Phase 1, load existing CSV
python prune.py --dataset cifar10 --arch resnet18 --prune_fraction 0.7 \
    --rho 2 --fine_tune_epochs 40 --device cuda:0 --skip_sensitivity
```

Note: `--rho` and `--num_clusters` must match the original run when using `--skip_sensitivity`, as they determine the CSV filename.

---

## Output Files

All outputs are written to `outputs/` and never touch the `models/` directory containing your pretrained checkpoints.

For a run with `--arch resnet18 --dataset cifar10 --prune_fraction 0.7 --rho 2 --fine_tune_epochs 40`:

| File | Description |
|------|-------------|
| `outputs/logs/fgsp_resnet18_cifar10_rho2_K20_ft40e_07PR_sensitivity.csv` | Per-cluster sensitivity scores from Phase 1 |
| `outputs/models/fgsp_resnet18_cifar10_rho2_K20_ft40e_07PR.pth` | Final pruned checkpoint with masks baked in |

The sensitivity CSV columns: `layer_name`, `cluster_id`, `sensitivity_score`, `neuron_indices`, `num_neurons`.

---

## Default Checkpoint Lookup

If `--checkpoint` is not passed, the script looks for the following filenames inside `--models_dir` (`models/` by default):

| arch | dataset | Expected filename |
|------|---------|-------------------|
| `resnet18` | `cifar10` | `resnet18_c10_no_sam_100e.pth` |
| `resnet18` | `cifar100` | `resnet18_c100_no_sam_100e.pth` |
| `wrn28_10` | `cifar10` | `wide-resnet_c10_no_sam_40e.pth` |
| `wrn28_10` | `cifar100` | `wide-resnet_c100_no_sam_40e.pth` |
| `resnet56` | `cifar10` | `resnet56_c10_no_sam_100e.pth` |
| `preact_resnet18` | `cifar10` | `preact_resnet18_c10_no_sam_100e.pth` |

To use a differently named checkpoint, pass `--checkpoint models/your_model.pth`.

---

## File Structure

```
fgsp/
├── prune.py              # main entry point
├── architectures.py      # ResNet18, WRN-28-10, ResNet56, PreActResNet18
├── utils.py              # load_dataset, normalize_cifar, normalize_cifar100
├── train.py              # standard / SAM training script
├── fgsp/
│   ├── analyzer.py       # Phase 1: clustered SAM sensitivity analysis
│   ├── pruner.py         # Phase 2: structured pruning + fine-tuning
│   └── models.py         # model factory + checkpoint lookup
└── requirements.txt
```