# import sys
# sys.path.append('./distributed-learning/')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import time

from consensus_simple.mixer import Mixer
from consensus_simple.utils import *
from consensus_simple.agent import Agent
from consensus_simple.statistic_collector import StatisticCollector

SEED = 42


def get_cifar10_train_loaders(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=args['dataset_dir'],
                                            train=True, download=True,
                                            transform=transform_train)

    agent_list = []
    dataset_sizes = []
    for agent_name, size in args['dataset_sizes'].items():
        agent_list.append(agent_name)
        dataset_sizes.append(size)

    subsets = torch.utils.data.random_split(trainset,
                                            lengths=dataset_sizes,
                                            generator=torch.Generator().manual_seed(SEED))

    train_loaders = {}
    for agent_name, subset in zip(agent_list, subsets):
        train_loaders[agent_name] = torch.utils.data.DataLoader(subset,
                                                                batch_size=args['train_batch_size'],
                                                                shuffle=True,
                                                                num_workers=2)
    return train_loaders


def get_cifar10_test_loader(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root=args['dataset_dir'],
                                           train=False, download=False,
                                           transform=transform_test)

    return torch.utils.data.DataLoader(testset, batch_size=args['test_batch_size'], shuffle=False, num_workers=2)


def get_resnet20_models(args):
    topology = args['topology']
    models = {}

    main_state_dict = None

    for agent in topology:
        models[agent] = args['model']()
        if args['equalize_start_params']:
            if main_state_dict is None:
                main_state_dict = models[agent].state_dict()
            else:
                models[agent].load_state_dict(main_state_dict)

        models[agent] = models[agent].cuda()
        models[agent] = torch.nn.DataParallel(models[agent], device_ids=range(torch.cuda.device_count()))

    return models


def get_criterion():
    return nn.CrossEntropyLoss().cuda()


def get_optimizer(args, model):
    return torch.optim.SGD(model.parameters(),
                           lr=args['lr'],
                           momentum=args['momentum'],
                           weight_decay=args['weight_decay'])


def get_lr_scheduler(optimizer, lr_schedule):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)


def save_params_statistics(network, stats):
    # variation
    agents_params = {agent_name: agent.get_flatten_params() for agent_name, agent in network.items()}
    params = np.array(list(agents_params.values()))
    max_var = np.linalg.norm(params.std(axis=0) / params.mean(axis=0), ord=np.inf)
    stats.add('coef_of_var', max_var)

    # deviation (L_1, L_2, L_inf)
    avg_params = params.mean(axis=0)
    deviation_params = {agent_name: p - avg_params for agent_name, p in agents_params.items()}
    stats.add('param_deviation_L1',
              {agent_name: np.linalg.norm(p, ord=1) for agent_name, p in deviation_params.items()})
    stats.add('param_deviation_L2',
              {agent_name: np.linalg.norm(p, ord=2) for agent_name, p in deviation_params.items()})
    stats.add('param_deviation_Linf',
              {agent_name: np.linalg.norm(p, ord=np.inf) for agent_name, p in deviation_params.items()})
    stats.dump_to_file()


def main(args):
    elapsed_time = 0
    start_time = time.time()

    required_args = [
        'dataset_name',
        'dataset_dir',
        'stat_path',
        'train_batch_size',
        'test_batch_size',
        'lr',
        'momentum',
        'weight_decay',
        'lr_schedule',
        'topology',
        'n_agents',
        'equalize_start_params',
        'use_lsr',
        'num_epochs',
        'train_freqs',
        'consensus_freqs',
        'stat_freq',
        'dataset_sizes',
        'model',
    ]
    for arg in required_args:
        if arg not in args:
            ValueError('Argument {} must be in args'.format(arg))

    topology = args['topology']
    logger = args['logger']
    consensus_freqs = args['consensus_freqs']
    train_freqs = args['train_freqs']

    logger.info('START with args \n{}'.format(args))

    # preparing
    test_loader = get_cifar10_test_loader(args)
    logger.info('Test loader with length {} successfully prepared'.format(len(test_loader)))

    train_loaders = get_cifar10_train_loaders(args)
    logger.info('Train loaders successfully prepared')

    models = get_resnet20_models(args)
    logger.info('{} Models successfully prepared'.format(len(models)))

    global_statistic = StatisticCollector('global_statistic',
                                          logger=logger,
                                          save_path=args['stat_path'] / 'global_statistic')
    logger.info('StatisticCollector successfully prepared')

    mixer = Mixer(topology, logger)
    logger.info('Mixer successfully prepared')

    elapsed_time += time.time() - start_time
    logger.info('Preparing took {}:{:02d}:{:02d}'.format(*get_hms(elapsed_time)))

    network = {}
    for agent_name in topology:
        model = models[agent_name]
        optimizer = get_optimizer(args, model)
        criterion = get_criterion()
        lr_scheduler = get_lr_scheduler(optimizer, lr_schedule=args['lr_schedule'])

        network[agent_name] = Agent(name=agent_name,
                                    model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    lr_scheduler=lr_scheduler,
                                    train_loader=train_loaders[agent_name],
                                    test_loader=test_loader,
                                    stats=StatisticCollector(agent_name,
                                                             logger=logger,
                                                             save_path=args['stat_path'] / str(agent_name)),
                                    train_freq=train_freqs[agent_name],
                                    stat_freq=args['stat_freq'])

    if args['num_epochs'] > 200:
        iterations = args['num_epochs']
    else:
        iterations = args['num_epochs']*max([len(loader) for loader in train_loaders.values()])
    # start training
    start_time = time.time()
    for it in range(1, iterations + 1):

        for agent_name, agent in network.items():
            agent.make_iteration()

        # passing parameters of neighbors
        for agent_name, agent in network.items():
            if it % consensus_freqs[agent_name] == 0:
                for neighbor_name in topology[agent_name]:
                    agent.set_consensus_params(neighbor_name, network[neighbor_name].get_flatten_params())

        for agent_name, agent in network.items():
            if it % consensus_freqs[agent_name] == 0:
                params = agent.get_params_for_averaging()
                params = mixer.mix(params, agent=agent_name)
                agent.load_flatten_params_to_model(params)

        if it % args['stat_freq'] == 0:
            logger.info('Iteration: {}'.format(it))
            save_params_statistics(network=network, stats=global_statistic)

            iteration_time = time.time() - start_time
            global_statistic.add('iteration_time', iteration_time)
            global_statistic.dump_to_file()
            elapsed_time += iteration_time
            logger.info('Elapsed time : {}:{:02d}:{:02d}'.format(*get_hms(elapsed_time)))
            start_time = time.time()

    logger.info('FINISH')


if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    main({})
