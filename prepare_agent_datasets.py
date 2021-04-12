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


def get_train_dataset():
    return __train_dataset


def get_val_dataset():
    return __val_dataset


def get_agent_val_loader(token, workers=0):
    # workers > 0 might cause deadlock
    return torch.utils.data.DataLoader(
        get_val_dataset(),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)


def get_agent_train_loader(token, batch_size, workers=0):
    # workers > 0 might cause deadlock
    inds = torch.load('data/{}/inds.torch'.format(token))
    ds = get_train_dataset()
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, indices=inds),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )


def prepare_agent_datasets(n):
    train_dataset = get_train_dataset()

    inds = np.arange(len(train_dataset) // n * n)
    np.random.shuffle(inds)
    split = np.split(inds, n)

    for token in range(n):
        print(f'{len(split[token])} samples for agent {token}')
        os.makedirs('data/{}/'.format(token), exist_ok=True)
        torch.save(split[token], 'data/{}/inds.torch'.format(token))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents-count', '-n', required=True, type=int)
    args = parser.parse_args()
    prepare_agent_datasets(args.agents_count)
