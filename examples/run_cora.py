import argparse

import numpy as np

import torch

from duet.data import load_dataset

from duet.methods import METHODS, RunConfig

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='cora')

    parser.add_argument('--device', default='cuda')

    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])

    parser.add_argument('--eps_x', type=float, default=1.0)

    parser.add_argument('--eps_a', type=float, default=8.0)

    parser.add_argument('--mechanism', default='mbm')

    parser.add_argument('--epochs', type=int, default=200)

    parser.add_argument('--patience', type=int, default=40)

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    methods = ['Vanilla', 'Ours']

    print('device:', device, 'dataset:', args.dataset, 'mech:', args.mechanism)

    for name in methods:

        accs = []

        for seed in args.seeds:

            data = load_dataset(args.dataset, seed=seed)

            cfg = RunConfig(mechanism=args.mechanism, eps_x=args.eps_x, eps_a=args.eps_a, device=device, model_name='gcn', lite=True, epochs=args.epochs, patience=args.patience, lr=0.01, k=8, lambda_f=0.05, dropout=0.5, weight_decay=0.0005, hidden=32)

            acc = METHODS[name](data, cfg, seed=seed)['test_acc'] * 100.0

            accs.append(acc)

            print(f'{name} seed{seed}: {acc:.2f}')

        print(f'{name}: {np.mean(accs):.2f}+/-{np.std(accs):.2f}')

if __name__ == '__main__':

    main()
