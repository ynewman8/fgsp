# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# import argparse
# from time import time

# from utils import *
# from model import PreActResNet18, WRN28_10, DeiT, VGG
# from sam import SAM, ASAM, ESAM


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--fname', type=str, required=True)
#     parser.add_argument('--model', type=str, default='PreActResNet18', choices=['PRN', 'WRN', 'DeiT', 'VGG'])
#     parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet-200'])
#     parser.add_argument('--epochs', default=100, type=int)
#     parser.add_argument('--max-lr', default=0.1, type=float)
#     parser.add_argument('--opt', default='SGD', choices=['Adam', 'SGD'])
#     parser.add_argument('--sam', default='NO', choices=['SAM', 'ASAM', 'ESAM', 'NO'])
#     parser.add_argument('--batch-size', default=128, type=int)
#     parser.add_argument('--device', default=0, type=int)
#     parser.add_argument('--adv', action='store_true')
#     parser.add_argument('--rho', default=0.05, type=float)  # for SAM

#     parser.add_argument('--norm', default='linf', choices=['linf', 'l2'])
#     parser.add_argument('--train-eps', default=8., type=float)
#     parser.add_argument('--train-alpha', default=2., type=float)
#     parser.add_argument('--train-step', default=5, type=int)

#     parser.add_argument('--test-eps', default=1., type=float)
#     parser.add_argument('--test-alpha', default=0.5, type=float)
#     parser.add_argument('--test-step', default=5, type=int)
#     return parser.parse_args()


# args = get_args()


# def lr_schedule(epoch):
#     if epoch < args.epochs * 0.75:
#         return args.max_lr
#     elif epoch < args.epochs * 0.9:
#         return args.max_lr * 0.1
#     else:
#         return args.max_lr * 0.01


# if __name__ == '__main__':
#     dataset = args.dataset
#     device = f'cuda:{args.device}'
#     model_name = args.model
#     label_dim = {'cifar10': 10, 'cifar100': 100, 'tiny-imagenet-200': 200}[dataset]
#     model = {'PRN': PreActResNet18(label_dim), 'WRN': WRN28_10(label_dim), 'DeiT': DeiT(label_dim), 'VGG': VGG(label_dim)}[model_name].to(
#         device)
#     train_loader, test_loader = load_dataset(dataset, args.batch_size)
#     params = model.parameters()
#     criterion = nn.CrossEntropyLoss()
    
#     if args.sam == 'NO':
#         if args.opt == 'SGD':
#             opt = torch.optim.SGD(params, lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
#         elif args.opt == 'Adam':
#             opt = torch.optim.Adam(params, lr=args.max_lr, weight_decay=5e-4)
#         else:
#             raise "Invalid optimizer"
#     else:
#         if args.sam == 'SAM':
#             base_opt = torch.optim.SGD
#             opt = SAM(params, base_opt, lr=args.max_lr, momentum=0.9, weight_decay=5e-4, rho=args.rho)
#         elif args.sam == 'ASAM':
#             base_opt = torch.optim.SGD(params, lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
#             opt = ASAM(base_opt, model, rho=args.rho)
#         elif args.sam == 'ESAM':
#             base_opt = torch.optim.SGD(model.parameters(), lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
#             opt = ESAM(params, base_opt, rho=args.rho)
#         else:
#             raise "Invalid SAM optimizer"

#     normalize = \
#     {'cifar10': normalize_cifar, 'cifar100': normalize_cifar100, 'tiny-imagenet-200': normalize_tinyimagenet}[dataset]

#     all_log_data = []
#     train_pgd = PGD(args.train_step, args.train_alpha / 255., args.train_eps / 255., args.norm, False, normalize)
#     test_pgd = PGD(args.test_step, args.test_alpha / 255., args.test_eps / 255., args.norm, False, normalize)

#     for epoch in range(args.epochs):
#         start_time = time()
#         log_data = [0, 0, 0, 0, 0, 0]  # train_loss, train_acc, test_loss, test_acc, test_robust_loss, test_robust
#         # train
#         model.train()
#         lr = lr_schedule(epoch)
#         if args.sam == 'ASAM':
#             opt.optimizer.param_groups[0].update(lr=lr)
#         else:
#             opt.param_groups[0].update(lr=lr)
#         for x, y in train_loader:
#             x, y = x.to(device), y.to(device)
#             if args.adv:
#                 delta = train_pgd.perturb(model, x, y)
#             else:
#                 delta = torch.zeros_like(x).to(x.device)

#             if args.sam == 'NO':
#                 output = model(normalize(x + delta))
#                 loss = criterion(output, y)
#                 opt.zero_grad()
#                 loss.backward()
#                 opt.step()

#             else:
#                 if args.sam == 'SAM':
#                     output = model(normalize(x + delta))
#                     loss = criterion(output, y)
#                     loss.backward()
#                     opt.first_step(zero_grad=True)
#                     output_2 = model(normalize(x + delta))
#                     criterion(output_2, y).backward()
#                     opt.second_step(zero_grad=True)
#                 elif args.sam == 'ASAM':
#                     output = model(normalize(x + delta))
#                     loss = criterion(output, y)
#                     loss.backward()
#                     opt.ascent_step()
#                     output_2 = model(normalize(x + delta))
#                     criterion(output_2, y).backward()
#                     opt.descent_step()
#                 elif args.sam == 'ESAM':
#                     def defined_backward(loss):
#                         loss.backward()
#                     paras = [normalize(x + delta), y, criterion, model, defined_backward]
#                     opt.paras = paras
#                     opt.step()
#                     output, loss = opt.returnthings

#             log_data[0] += (loss * len(y)).item()
#             log_data[1] += (output.max(1)[1] == y).float().sum().item()

#         # test
#         model.eval()
#         for x, y in test_loader:
#             x, y = x.to(device), y.to(device)
#             # clean
#             output = model(normalize(x)).detach()
#             loss = criterion(output, y)

#             log_data[2] += (loss * len(y)).item()
#             log_data[3] += (output.max(1)[1] == y).float().sum().item()
#             delta = test_pgd.perturb(model, x, y)
#             output = model(normalize(x + delta)).detach()
#             loss = criterion(output, y)

#             log_data[4] += (loss * len(y)).item()
#             log_data[5] += (output.max(1)[1] == y).float().sum().item()

#         log_data = np.array(log_data)
#         num_train = 60000 if 'cifar' in dataset else 100000
#         num_test = 10000 if 'cifar' in dataset else 10000
#         log_data[0] /= num_train
#         log_data[1] /= num_train
#         log_data[2] /= num_test
#         log_data[3] /= num_test
#         log_data[4] /= num_test
#         log_data[5] /= num_test
#         all_log_data.append(log_data)

#         print(f'Epoch {epoch}:\t', log_data, f'\tTime {time() - start_time:.1f}s')
#         save_path = '{dataset}_models/{fname}.pth'
#         torch.save(model.state_dict(), save_path.format(dataset=dataset, fname=args.fname))

#     all_log_data = np.stack(all_log_data, axis=0)

#     df = pd.DataFrame(all_log_data)
#     df.to_csv(f'logs/{args.fname}.csv')

#     plt.plot(all_log_data[:, [2, 4]])
#     plt.grid()
#     # plt.title(f'{dataset} {args.opt}{" adv" if args.adv else ""} Loss', fontsize=16)
#     plt.legend(['clean', 'robust'], fontsize=16)
#     plt.savefig(f'figs/{args.fname}_loss.png', dpi=200)
#     plt.clf()

#     plt.plot(all_log_data[:, [3, 5]])
#     plt.grid()
#     # plt.title(f'{dataset} {args.opt}{" adv" if args.adv else ""} Acc', fontsize=16)
#     plt.legend(['clean', 'robust'], fontsize=16)
#     plt.savefig(f'figs/{args.fname}_acc.png', dpi=200)
#     plt.clf()


#### ABOVE IS THE ORIGINAL CODE (WITH SOME MODIFICATIONS), BELOW IS THE CODE FOR PERFORMING SAM ON PRUNED MODELS
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from time import time

from utils import load_dataset, normalize_cifar, normalize_cifar100, PGD
from model import PreActResNet18, WRN28_10, DeiT, VGG, ResNet56, ResNet18
from sam import SAM, ASAM, ESAM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True, help='Output filename for model and logs')
    parser.add_argument('--model', type=str, default='PreActResNet18', choices=['PRN', 'WRN', 'DeiT',  'RN56', 'ResNet56', 'VGG', 'ResNet18'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--max-lr', default=0.1, type=float)
    parser.add_argument('--opt', default='SGD', choices=['Adam', 'SGD'])
    parser.add_argument('--sam', default='NO', choices=['SAM', 'ASAM', 'ESAM', 'NO'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--rho', default=0.05, type=float)
    parser.add_argument('--train-eps', default=8., type=float)
    parser.add_argument('--train-alpha', default=2., type=float)
    parser.add_argument('--train-step', default=5, type=int)
    parser.add_argument('--test-eps', default=1., type=float)
    parser.add_argument('--test-alpha', default=0.5, type=float)
    parser.add_argument('--test-step', default=5, type=int)
    parser.add_argument('--norm', default='linf', choices=['linf', 'l2'])
    parser.add_argument('--load-pruned', type=str, default=None, help='Path to pruned model to load (e.g., models/v3_combined_kmeans_5percent_clusters_XXPR.pth)')
    return parser.parse_args()

def lr_schedule(epoch, total_epochs, max_lr):
    if epoch < total_epochs * 0.75:
        return max_lr
    elif epoch < total_epochs * 0.9:
        return max_lr * 0.1
    else:
        return max_lr * 0.01

if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    device = f'cuda:{args.device}'
    model_name = args.model
    label_dim = {'cifar10': 10, 'cifar100': 100}[dataset]
    # model = {'PRN': PreActResNet18(label_dim), 'WRN': WRN28_10(label_dim), 'DeiT': DeiT(label_dim), 'VGG': VGG(label_dim)}[model_name].to(device)
    MODEL_ZOO = {
    'PRN': PreActResNet18,
    'WRN': WRN28_10,
    'RN56': ResNet56,        
    'ResNet56': ResNet56,    
    'DeiT': DeiT,
    'VGG': VGG,
    'ResNet18': ResNet18,
}

    model = MODEL_ZOO[model_name](label_dim).to(device)

    # Load pruned model if specified
    if args.load_pruned:
        state = torch.load(args.load_pruned, map_location=device)
        model.load_state_dict(state)
        print(f'Loaded pruned model from {args.load_pruned}')

    train_loader, test_loader = load_dataset(dataset, args.batch_size)
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()

    if args.sam == 'NO':
        if args.opt == 'SGD':
            opt = torch.optim.SGD(params, lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
        elif args.opt == 'Adam':
            opt = torch.optim.Adam(params, lr=args.max_lr, weight_decay=5e-4)
        else:
            raise ValueError("Invalid optimizer")
    else:
        if args.sam == 'SAM':
            base_opt = torch.optim.SGD
            opt = SAM(params, base_opt, lr=args.max_lr, momentum=0.9, weight_decay=5e-4, rho=args.rho)
        elif args.sam == 'ASAM':
            base_opt = torch.optim.SGD(params, lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
            opt = ASAM(base_opt, model, rho=args.rho)
        elif args.sam == 'ESAM':
            base_opt = torch.optim.SGD(model.parameters(), lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
            opt = ESAM(params, base_opt, rho=args.rho)
        else:
            raise ValueError("Invalid SAM optimizer")

    normalize = {'cifar10': normalize_cifar, 'cifar100': normalize_cifar100}[dataset]

    all_log_data = []
    train_pgd = PGD(args.train_step, args.train_alpha / 255., args.train_eps / 255., args.norm, False, normalize)
    test_pgd = PGD(args.test_step, args.test_alpha / 255., args.test_eps / 255., args.norm, False, normalize)

    for epoch in range(args.epochs):
        start_time = time()
        log_data = [0, 0, 0, 0, 0, 0]  # train_loss, train_acc, test_loss, test_acc, test_robust_loss, test_robust
        # Train
        model.train()
        lr = lr_schedule(epoch, args.epochs, args.max_lr)
        opt.param_groups[0].update(lr=lr)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.adv:
                delta = train_pgd.perturb(model, x, y)
            else:
                delta = torch.zeros_like(x).to(x.device)

            if args.sam == 'NO':
                output = model(normalize(x + delta))
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                if args.sam == 'SAM':
                    output = model(normalize(x + delta))
                    loss = criterion(output, y)
                    loss.backward()
                    opt.first_step(zero_grad=True)
                    output_2 = model(normalize(x + delta))
                    criterion(output_2, y).backward()
                    opt.second_step(zero_grad=True)
                elif args.sam == 'ASAM':
                    output = model(normalize(x + delta))
                    loss = criterion(output, y)
                    loss.backward()
                    opt.ascent_step()
                    output_2 = model(normalize(x + delta))
                    criterion(output_2, y).backward()
                    opt.descent_step()
                elif args.sam == 'ESAM':
                    def defined_backward(loss):
                        loss.backward()
                    paras = [normalize(x + delta), y, criterion, model, defined_backward]
                    opt.paras = paras
                    opt.step()
                    output, loss = opt.returnthings

            log_data[0] += (loss * len(y)).item()
            log_data[1] += (output.max(1)[1] == y).float().sum().item()

        # Test
        model.eval()
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            # Clean
            output = model(normalize(x)).detach()
            loss = criterion(output, y)
            log_data[2] += (loss * len(y)).item()
            log_data[3] += (output.max(1)[1] == y).float().sum().item()
            delta = test_pgd.perturb(model, x, y)
            output = model(normalize(x + delta)).detach()
            loss = criterion(output, y)
            log_data[4] += (loss * len(y)).item()
            log_data[5] += (output.max(1)[1] == y).float().sum().item()

        log_data = np.array(log_data)
        num_train = 60000 if 'cifar' in dataset else 100000
        num_test = 10000 if 'cifar' in dataset else 10000
        log_data[0] /= num_train
        log_data[1] /= num_train
        log_data[2] /= num_test
        log_data[3] /= num_test
        log_data[4] /= num_test
        log_data[5] /= num_test
        all_log_data.append(log_data)

        print(f'Epoch {epoch}:\t', log_data, f'\tTime {time() - start_time:.1f}s')
        save_path = f'models/{args.fname}.pth'
        torch.save(model.state_dict(), save_path)
        print(f'Saved model to {save_path}')

    all_log_data = np.stack(all_log_data, axis=0)
    df = pd.DataFrame(all_log_data)
    df.to_csv(f'logs/{args.fname}.csv')

    plt.plot(all_log_data[:, [2, 4]])
    plt.grid()
    plt.legend(['clean', 'robust'], fontsize=16)
    plt.savefig(f'figs/{args.fname}_loss.png', dpi=200)
    plt.clf()

    plt.plot(all_log_data[:, [3, 5]])
    plt.grid()
    plt.legend(['clean', 'robust'], fontsize=16)
    plt.savefig(f'figs/{args.fname}_acc.png', dpi=200)
    plt.clf()

