import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy
import os
import shutil
import torch
import numpy as np

def prepare_agent_datasets():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    inds = np.arange(len(train_dataset))
    np.random.shuffle(inds)
    split = np.split(inds, 10)

    for token in range(10):
        X = []
        Y = []
        for ind in split[token]:
            x, y = train_dataset[ind]
            X.append(x)
            Y.append(y)
        print('{} samples for agent {}'.format(len(Y), token))
        ds = torch.utils.data.TensorDataset(torch.stack(X),
                                            torch.tensor(numpy.array(Y), dtype=torch.long))
        os.makedirs('data/{}/'.format(token), exist_ok=True)
        torch.save(ds, 'data/{}/dataset.torch'.format(token))

if __name__ == '__main__':
    prepare_agent_datasets()