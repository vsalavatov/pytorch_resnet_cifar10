# import sys
# sys.path.append('./distributed-learning/')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import time
import pickle

from consensus_simple.mixer import Mixer
from consensus_simple.weighted_mixer import WeightedMixer
from consensus_simple.utils import *
from consensus_simple.agent import Agent
from consensus_simple.statistic_collector import StatisticCollector

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


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


def get_lr_scheduler(optimizer, lr_schedule, weight=None):
    if weight:
        def schedule(x):
            return weight * lr_schedule(x)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
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


def main(args):
    elapsed_time = 0
    start_time = time.time()

    required_args = [
        'dataset_name',
        'dataset_dir',
        'save_path',
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
            raise ValueError('Argument {} must be in args'.format(arg))
    if 'weights' in args and not isinstance(args['consensus_freqs'], int):
        raise ValueError('If "weights" in args, consensus_freqs must be int.')
    if 'weights' in args:
        for arg in ['consensus_lr_schedule', 'consensus_lr']:
            if arg not in args:
                raise ValueError('If "weights" in args, argument {} must be in args too'.format(arg))

    topology = args['topology']
    logger = args['logger']
    consensus_freqs = args['consensus_freqs']
    train_freqs = args['train_freqs']

    logger.info('START with args \n{}'.format(args))

    if 'checkpoint_path' in args and args['checkpoint_path']:
        checkpoint = torch.load(args['checkpoint_path'] / 'global_statistic.th')
        global_statistic = StatisticCollector.load_from_file(args['save_path'] / 'global_statistic')
        logger.info('StatisticCollector successfully loaded from checkpoint')
        start_iteration = checkpoint['iteration'] + 1
        mixer = checkpoint['mixer']
        logger.info('Mixer successfully loaded from checkpoint')
    else:
        if 'weights' in args:
            mixer = WeightedMixer(topology,
                                  logger,
                                  args['weights'],
                                  args['consensus_lr_schedule'],
                                  args['consensus_lr'])
        else:
            mixer = Mixer(topology, logger)
        logger.info('Mixer successfully prepared')

        global_statistic = StatisticCollector('global_statistic',
                                              logger=logger,
                                              save_path=args['save_path'] / 'global_statistic')
        logger.info('StatisticCollector successfully prepared')
        start_iteration = 1

    # preparing
    test_loader = get_cifar10_test_loader(args)
    logger.info('Test loader with length {} successfully prepared'.format(len(test_loader)))

    train_loaders = get_cifar10_train_loaders(args)
    logger.info('Train loaders successfully prepared')

    models = get_resnet20_models(args)
    logger.info('{} Models successfully prepared'.format(len(models)))

    elapsed_time += time.time() - start_time
    logger.info('Preparing took {}:{:02d}:{:02d}'.format(*get_hms(elapsed_time)))

    network = {}
    for agent_name in topology:
        model = models[agent_name]
        optimizer = get_optimizer(args, model)
        criterion = get_criterion()
        if 'lsr_weights' in args:
            lr_scheduler = get_lr_scheduler(optimizer, lr_schedule=args['lr_schedule'],
                                            weight=args['lsr_weights'][agent_name])
        else:
            lr_scheduler = get_lr_scheduler(optimizer, lr_schedule=args['lr_schedule'])
        stats = StatisticCollector(agent_name,
                                   logger=logger,
                                   save_path=args['save_path'] / str(agent_name))

        if 'checkpoint_path' in args and args['checkpoint_path']:
            checkpoint = torch.load(args['checkpoint_path'] / '{}.th'.format(agent_name))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            stats = StatisticCollector.load_from_file(args['save_path'] / str(agent_name))

        network[agent_name] = Agent(name=agent_name,
                                    model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    lr_scheduler=lr_scheduler,
                                    train_loader=train_loaders[agent_name],
                                    test_loader=test_loader,
                                    stats=stats,
                                    train_freq=train_freqs[agent_name],
                                    stat_freq=args['stat_freq'])

        if 'checkpoint_path' in args and args['checkpoint_path']:
            network[agent_name]._buffer_stat = checkpoint['_buffer_stat']
            logger.info('Agent {} successfully loaded from checkpoint'.format(agent_name))
        else:
            logger.info('Agent {} successfully prepared'.format(agent_name))

    if args['num_epochs'] > 200:
        iterations = args['num_epochs']
    else:
        iterations = args['num_epochs'] * max([len(loader) for loader in train_loaders.values()])
    # start training
    logger.info('Training started from {} iteration'.format(start_iteration))
    start_time = time.time()
    for it in range(start_iteration, iterations + 1):

        for agent_name, agent in network.items():
            agent.make_iteration()

        if 'weights' in args:
            if it % consensus_freqs == 0:
                agents_params = {}
                for agent_name, agent in network.items():
                    agents_params[agent_name] = agent.get_flatten_params()
                agents_params = mixer.mix(agents_params, iteration=it)
                for agent_name, agent in network.items():
                    agent.load_flatten_params_to_model(agents_params[agent_name])
        else:
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

        if it % args['stat_freq'] == 0 or it == iterations:
            logger.info('Iteration: {} / {}'.format(it, iterations))
            for agent_name, agent in network.items():
                agent.test()
            save_params_statistics(network=network, stats=global_statistic)

            iteration_time = time.time() - start_time
            global_statistic.add('iterations_time', iteration_time).dump_to_file()
            elapsed_time += iteration_time
            logger.info('Elapsed time : {}:{:02d}:{:02d}'.format(*get_hms(elapsed_time)))
            start_time = time.time()

            save_checkpoint({
                'iteration': it,
                'mixer': mixer,
            }, filename=args['save_path'] / 'global_statistic.th')

            for agent_name, agent in network.items():
                save_checkpoint({
                    'model': agent.model.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'lr_scheduler': agent.lr_scheduler.state_dict(),
                    '_buffer_stat': agent._buffer_stat,
                }, filename=args['save_path'] / '{}.th'.format(agent_name))
                agent.stats.dump_to_file()

    logger.info('FINISH')


if __name__ == '__main__':
    pass
