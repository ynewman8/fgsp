import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import SVHN
from sklearn.metrics import roc_curve, auc

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils import load_dataset, normalize_cifar, normalize_cifar100, PGD
from fgsp.models import build_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the pruned model .pth')
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'wrn28_10', 'resnet56', 'preact_resnet18'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--device', default=None, help='e.g. cuda:0')
    parser.add_argument('--output_csv', default='eval_results.csv', type=str)
    return parser.parse_args()


def pixel_attack(model, x, y, device, num_pixels=1, normalize=None):
    x_adv = x.clone().detach().to(device)
    batch_size, channels, height, width = x_adv.shape
    for i in range(batch_size):
        pixel_indices = torch.randperm(height * width)[:num_pixels]
        row_indices = (pixel_indices // width).long()
        col_indices = (pixel_indices % width).long()
        for j in range(num_pixels):
            x_adv[i, :, row_indices[j], col_indices[j]] = torch.rand(channels).to(device)
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv

def load_cifar_c(dataset_name='cifar10', corruption='gaussian_noise', batch_size=1024):
    if dataset_name == 'cifar10':
        data_folder = './datasets/cifar-10-c/CIFAR-10-C'
    elif dataset_name == 'cifar100':
        data_folder = './datasets/cifar-100-c/CIFAR-100-C'
    else:
        return None

    file_path = os.path.join(data_folder, f'{corruption}.npy')
    if not os.path.isfile(file_path):
        print(f"{file_path} not found. Check that the dataset corresponds to {dataset_name}-C is downloaded and extracted.")
        return None
    try:
        images = np.load(file_path)
        images = images[:10000]  # First 10,000 images (severity level 1)
        images = torch.from_numpy(images).float()
        if images.ndim == 3:
            images = images.unsqueeze(1).repeat(1, 3, 1, 1)
        elif images.ndim == 4 and images.shape[3] == 3:
            images = images.permute(0, 3, 1, 2)
        images = images / 255.0
        
        if dataset_name == 'cifar10':
            base_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        else:
            base_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms.ToTensor())
            
        labels = torch.tensor(base_set.targets)
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader
    except Exception as e:
        print(f"Error loading {dataset_name}-C for corruption '{corruption}': {e}")
        return None

def evaluate_clean(model, loader, normalize_fn, device='cuda'):
    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(normalize_fn(x))
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total

def evaluate_label_noise(model, loader, normalize_fn, noise_fraction=0.2, device='cuda'):
    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            noisy_y = y.clone()
            num_noisy = int(noisy_y.size(0) * noise_fraction)
            if num_noisy > 0:
                indices = torch.randperm(noisy_y.size(0))[:num_noisy]
                if hasattr(model, 'fc'):
                    num_classes = model.fc.out_features
                elif hasattr(model, 'linear'):
                    num_classes = model.linear.out_features
                elif hasattr(model, 'classifier'):
                    last_layer = model.classifier[-1]
                    if isinstance(last_layer, nn.Linear):
                        num_classes = last_layer.out_features
                    else:
                        raise ValueError("Model final layer in classifier not recognized")
                else:
                    raise ValueError("Model final layer not recognized")
                noisy_y[indices] = torch.randint(0, num_classes, (num_noisy,), device=device)
            outputs = model(normalize_fn(x))
            correct += (outputs.argmax(dim=1) == noisy_y).sum().item()
            total += len(y)
    return correct / total

def compute_distribution_shift_metric(corruption_results):
    accuracies = list(corruption_results.values())
    return np.std(accuracies) if len(accuracies) > 0 else None

def evaluate_pixel_attack(model, loader, normalize_fn, num_pixels=1, device='cuda'):
    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_adv = pixel_attack(model, x, y, device=device, num_pixels=num_pixels, normalize=normalize_fn)
            outputs = model(normalize_fn(x_adv))
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total

def evaluate_attacks(model, loader, attacks, normalize_fn, device='cuda'):
    attack_accs = []
    for attack in attacks:
        total, correct = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            delta = attack.perturb(model, x, y)
            outputs = model(normalize_fn(x + delta))
            correct += (outputs.argmax(dim=1) == y).sum().item()
            total += len(y)
        attack_accs.append(correct / total)
    return attack_accs

def evaluate_cifarc(model, corruption_types, dataset_name='cifar10', normalize_fn=None, device='cuda', batch_size=1024):
    results = {}
    for corr in corruption_types:
        loader = load_cifar_c(dataset_name=dataset_name, corruption=corr, batch_size=batch_size)
        if loader is None:
            continue
        total, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                outputs = model(normalize_fn(x))
                correct += (outputs.argmax(dim=1) == y).sum().item()
                total += len(y)
        results[corr] = correct / total
    return results

def compute_mce(cifarc_results):
    if not cifarc_results: return None
    errors = [1 - acc for acc in cifarc_results.values()]
    return np.mean(errors)

def compute_v2_metric(clean_acc, cifarc_results):
    if not cifarc_results: return None
    avg_corr_acc = np.mean(list(cifarc_results.values()))
    return clean_acc - avg_corr_acc

def evaluate_fgsm(model, loader, normalize_fn, eps=8./255., device='cuda'):
    total, correct = 0, 0
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x.requires_grad = True
        outputs = model(normalize_fn(x))
        loss = nn.CrossEntropyLoss()(outputs, y)
        model.zero_grad()
        loss.backward()
        grad_sign = x.grad.sign()
        x_adv = x + eps * grad_sign
        x_adv = torch.clamp(x_adv, 0, 1)
        outputs_adv = model(normalize_fn(x_adv))
        correct += (outputs_adv.argmax(dim=1) == y).sum().item()
        total += len(y)
    return correct / total

def compute_roc(model, dataset_name, normalize_fn, batch_size=128, device='cuda'):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == 'cifar10':
        id_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        id_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    id_loader = DataLoader(id_set, batch_size=batch_size, shuffle=False)
    ood_set = SVHN(root='./data', split='test', download=True, transform=transform)
    ood_loader = DataLoader(ood_set, batch_size=batch_size, shuffle=False)
    
    scores = []
    labels = []
    with torch.no_grad():
        for x, _ in id_loader:
            x = x.to(device)
            outputs = model(normalize_fn(x))
            softmax = torch.softmax(outputs, dim=1)
            scores.extend(softmax.max(dim=1)[0].cpu().numpy())
            labels.extend(np.ones(len(x)))
        for x, _ in ood_loader:
            x = x.to(device)
            outputs = model(normalize_fn(x))
            softmax = torch.softmax(outputs, dim=1)
            scores.extend(softmax.max(dim=1)[0].cpu().numpy())
            labels.extend(np.zeros(len(x)))
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return {'roc_auc': roc_auc}

def main():
    args = get_args()
    
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = args.device

    print(f"Evaluating model checkpoint: {args.checkpoint}")
    
    num_classes = 10 if args.dataset == 'cifar10' else 100
    normalize_fn = normalize_cifar if args.dataset == 'cifar10' else normalize_cifar100
    
    _, test_loader = load_dataset(args.dataset, args.batch_size)

    # Build model using fgsp_repo builder
    model = build_model(args.arch, num_classes, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # Define Attacks
    PGD1  = PGD(10, 0.25/255., 1./255., 'linf', normalize=normalize_fn)
    PGD2  = PGD(10, 0.5/255., 2./255., 'linf', normalize=normalize_fn)
    PGD16 = PGD(10, 2./255., 16./255., 'l2', normalize=normalize_fn)
    PGD32 = PGD(10, 4./255., 32./255., 'l2', normalize=normalize_fn)
    attacks = [PGD1, PGD2, PGD16, PGD32]

    corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness', 'contrast',
        'elastic_transform', 'pixelate', 'jpeg_compression'
    ]

    print("Computing metrics...")
    clean_acc = evaluate_clean(model, test_loader, normalize_fn, device=device)
    attack_accs = evaluate_attacks(model, test_loader, attacks, normalize_fn, device=device)
    cifarc_results = evaluate_cifarc(model, corruption_types, dataset_name=args.dataset, normalize_fn=normalize_fn, device=device)
    mce = compute_mce(cifarc_results)
    rmce = compute_v2_metric(clean_acc, cifarc_results)
    pixel_acc = evaluate_pixel_attack(model, test_loader, normalize_fn, num_pixels=40, device=device)
    label_noise_acc = evaluate_label_noise(model, test_loader, normalize_fn, noise_fraction=0.2, device=device)
    dist_shift_std = compute_distribution_shift_metric(cifarc_results)
    fgsm_acc = evaluate_fgsm(model, test_loader, normalize_fn, eps=8./255., device=device)
    roc_metrics = compute_roc(model, args.dataset, normalize_fn, batch_size=args.batch_size, device=device)

    print(f"  Clean Accuracy: {clean_acc:.4f}")
    print("  PGD Attack Accuracies:")
    print(f"    PGD1: {attack_accs[0]:.4f}")
    print(f"    PGD2: {attack_accs[1]:.4f}")
    print(f"    PGD16: {attack_accs[2]:.4f}")
    print(f"    PGD32: {attack_accs[3]:.4f}")
    print(f"  FGSM Accuracy: {fgsm_acc:.4f}")
    print(f"  Pixel Attack Accuracy: {pixel_acc:.4f}")
    print(f"  Label Noise Accuracy: {label_noise_acc:.4f}")
    print(f"  Distribution Shift STD: {dist_shift_std if dist_shift_std is not None else 'N/A'}")
    if cifarc_results:
        print(f"  {args.dataset.upper()}-C Accuracies:")
        for corr, acc in cifarc_results.items():
            print(f"    {corr}: {acc:.4f}")
    print(f"  mCE: {mce if mce is not None else 'N/A'}")
    print(f"  RmCE: {rmce if rmce is not None else 'N/A'}")
    print(f"  ROC AUC (OOD detection): {roc_metrics['roc_auc']:.4f}")
    print("-" * 50)
    
    individual_corruptions = {corr: cifarc_results.get(corr, None) for corr in corruption_types}

    result = {
        'Model': os.path.basename(args.checkpoint),
        'Clean_Acc': clean_acc,
        'mCE': mce,
        'RmCE': rmce,
        'PGD1': attack_accs[0],
        'PGD2': attack_accs[1],
        'PGD16': attack_accs[2],
        'PGD32': attack_accs[3],
        'FGSM': fgsm_acc,
        'Pixel_Attack': pixel_acc,
        'Label_Noise': label_noise_acc,
        'Dist_Shift_STD': dist_shift_std,
        **individual_corruptions,
        'ROC_AUC': roc_metrics['roc_auc']
    }

    df = pd.DataFrame([result])
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to {args.output_csv}")

if __name__ == '__main__':
    main()
