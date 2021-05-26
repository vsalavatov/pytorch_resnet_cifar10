import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy
import os
import shutil
import torch
import numpy as np
import argparse

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

normalize_1d = transforms.Normalize(mean=[0],
                                    std=[256])

__train_dataset = None
__val_dataset = None


def load_dataset(name):
    global __train_dataset, __val_dataset
    if name == 'cifar10':
        __train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

        __val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    elif name == 'fashion_mnist':
        __train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_1d,
        ]), download=True)
        __val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        raise ValueError('This dataset not implemented')


def get_train_dataset():
    with open('data/dataset_name.txt', 'r') as f:
        name = f.readline()
        load_dataset(name)
    return __train_dataset


def get_val_dataset():
    with open('data/dataset_name.txt', 'r') as f:
        name = f.readline()
        load_dataset(name)
    return __val_dataset


def get_agent_val_loader(token, workers=0):
    return torch.utils.data.DataLoader(
        get_val_dataset(),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)


def get_agent_train_loader(token, batch_size, workers=0):
    inds = torch.load('data/{}/inds.torch'.format(token))
    ds = get_train_dataset()
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, indices=inds),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    ), len(inds)


def prepare_agent_datasets(n, selected_sizes=None, selected_nodes=None):
    train_dataset = get_train_dataset()

    if selected_nodes is None:
        inds = np.arange(len(train_dataset) // n * n)
        np.random.shuffle(inds)
        split = np.split(inds, n)
    else:
        default_size = (len(train_dataset) - sum(selected_sizes)) // (n - len(selected_nodes))
        split_id = [default_size for _ in range(n)]
        for i in range(len(selected_sizes)):
            split_id[selected_nodes[i]] = selected_sizes[i]
        inds = np.arange(sum(split_id))
        split_id = np.cumsum(split_id)
        split = np.split(inds, split_id)

    for token in range(n):
        print(f'{len(split[token])} samples for agent {token}')
        os.makedirs('data/{}/'.format(token), exist_ok=True)
        torch.save(split[token], 'data/{}/inds.torch'.format(token))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents-count', '-n', required=True, type=int)
    parser.add_argument('--sizes', nargs='*', type=int)
    parser.add_argument('--nodes', nargs='*', type=int)
    parser.add_argument('--name', choices=['cifar10', 'fashion_mnist'], type=str)

    args = parser.parse_args()

    with open('data/dataset_name.txt', 'w') as f:
        f.write(args.name)
    load_dataset(args.name)
    prepare_agent_datasets(args.agents_count, args.sizes, args.nodes)
