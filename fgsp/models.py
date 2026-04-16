"""
fgsp/models.py — Model factory using local architectures.py
"""
import os
import torch.nn as nn
from architectures import ResNet18, WRN28_10, ResNet56, PreActResNet18

ARCH_MAP = {
    'resnet18':        ResNet18,
    'wrn28_10':        WRN28_10,
    'resnet56':        ResNet56,
    'preact_resnet18': PreActResNet18,
}

DEFAULT_CKPT = {
    ('resnet18',        'cifar10'):  'resnet18_c10_no_sam_100e.pth',
    ('resnet18',        'cifar100'): 'resnet18_c100_no_sam_100e.pth',
    ('wrn28_10',        'cifar10'):  'wide-resnet_c10_no_sam_40e.pth',
    ('wrn28_10',        'cifar100'): 'wide-resnet_c100_no_sam_40e.pth',
    ('resnet56',        'cifar10'):  'resnet56_c10_no_sam_100e.pth',
    ('preact_resnet18', 'cifar10'):  'preact_resnet18_c10_no_sam_100e.pth',
}

def build_model(arch: str, num_classes: int, device: str) -> nn.Module:
    arch = arch.lower()
    if arch not in ARCH_MAP:
        raise ValueError(f"Unknown arch '{arch}'. Choose from: {list(ARCH_MAP.keys())}")
    return ARCH_MAP[arch](num_classes).to(device)

def default_checkpoint(arch: str, dataset: str, models_dir: str = 'models') -> str:
    key = (arch.lower(), dataset.lower())
    filename = DEFAULT_CKPT.get(key)
    if filename is None:
        raise ValueError(f"No default checkpoint for arch={arch}, dataset={dataset}. Pass --checkpoint explicitly.")
    return os.path.join(models_dir, filename)
