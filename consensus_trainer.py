'''
This is a plain consensus version modification on trainer.py
'''

import argparse
import asyncio
import os
import pickle
import sys
import time
import numpy as np
import resnet
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

sys.path.append('./distributed-learning/')

from model_statistics import ModelStatistics
from utils.consensus_tcp import ConsensusAgent
from prepare_agent_datasets import get_agent_train_loader, get_agent_val_loader
from consensus_master import TelemetryModelParameters, TelemetryAgentGeneralInfo

model_names = sorted(name for name in resnet.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))


def make_config_parser():
    parser = argparse.ArgumentParser(description='Proper ResNets for CIFAR10 in pytorch')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet20)')

    # Arguments for consensus:
    parser.add_argument('--agent-token', '-t', required=True, type=int)
    parser.add_argument('--agent-host', default='127.0.0.1', type=str)
    parser.add_argument('--agent-port', required=True, type=int)
    parser.add_argument('--init-leader', dest='init_leader', action='store_true')
    parser.add_argument('--master-host', default='127.0.0.1', type=str)
    parser.add_argument('--master-port', required=True, type=int)
    parser.add_argument('--enable-log', dest='logging', action='store_true')
    parser.add_argument('--total-agents', required=True, type=int)
    parser.add_argument('--debug-consensus', dest='debug', action='store_true')
    parser.add_argument('--use-prepared-data', dest='data_prepared', action='store_true')
    parser.add_argument('--consensus-freq', dest='consensus_frequency', type=int, default=1,
                        help='freq>0 -> do averaging <freq> times per batch, '
                             'freq<0 -> do averaging once per (-freq) batches')
    # parser.add_argument('--use-consensus-rounds', dest='use_consensus_rounds', action='store_true')
    # parser.add_argument('--consensus-rounds-precision', dest='consensus_rounds_precision', type=float, default=1e-4)
    parser.add_argument('--no-validation', dest='no_validation', action='store_true')
    parser.add_argument('--use-lsr', dest='use_lsr', action='store_true')
    parser.add_argument('--warmup', dest='warmup', default=0, type=int)
    parser.add_argument('--momentum-consensus', dest='momentum_consensus', action='store_true')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (d  efault: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use half-precision(16-bit) ')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)
    parser.add_argument('--save-every', dest='save_every',
                        help='Saves checkpoints at every specified number of epochs',
                        type=int, default=10)

    return parser


class ConsensusSpecific:
    def __init__(self, cfg):
        self.cfg = cfg
        self.agent = None
        self.agent_serve_task = None

        self.run_averaging_exec_count = 0
        self.batch_counter = 0

    def init_consensus(self):
        self.agent = ConsensusAgent(self.cfg.agent_token, self.cfg.agent_host, self.cfg.agent_port,
                                    self.cfg.master_host, self.cfg.master_port,
                                    debug=True if self.cfg.debug else False)
        self.agent_serve_task = asyncio.create_task(self.agent.serve_forever())
        print('{}: Created serving task'.format(self.cfg.agent_token))

    def stop_consensus(self):
        self.agent_serve_task.cancel()

    def dump_params(self, model, optimizer_manager=None):
        model_params = torch.cat([p.data.to(torch.float32).view(-1) for p in model.parameters()]).detach().clone().cpu().numpy()
        if optimizer_manager is None:
            return model_params
        else:
            optimizer_params = optimizer_manager.extract()
            return np.concatenate([model_params, optimizer_params])

    def load_params(self, model, params, optimizer_manager=None):
        used_params = 0
        for p in model.parameters():
            cnt_params = p.numel()
            p.data.copy_(torch.Tensor(params[used_params:used_params + cnt_params]).view(p.shape).to(p.dtype))
            used_params += cnt_params
        if optimizer_manager is not None:
            optimizer_manager.set(params[used_params:])

    async def run_averaging(self, model, optimizer=None):
        if optimizer is not None:
            optimizer_manager = MomentumBufferManager(optimizer)
        else:
            optimizer_manager = None
        if self.cfg.consensus_frequency < 0:
            if self.run_averaging_exec_count % (-self.cfg.consensus_frequency) == 0:
                params = self.dump_params(model, optimizer_manager)
                params = await self.agent.run_once(params)
                self.load_params(model, params, optimizer_manager)
        else:
            params = self.dump_params(model, optimizer_manager)
            for _ in range(self.cfg.consensus_frequency):
                params = await self.agent.run_once(params)
            self.load_params(model, params, optimizer_manager)
        self.run_averaging_exec_count += 1


class MomentumBufferManager:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.shapes = []
        self.sizes = []

    def extract(self):
        momentum_buffer_list = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.optimizer.state[p]
                    if 'momentum_buffer' not in state:
                        raise ValueError('Initialize momentum buffer before extract them')
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

        extracted_buf = []
        for buf in momentum_buffer_list:
            np_buf = buf.clone().detach().cpu().numpy()
            self.shapes.append(np_buf.shape)
            self.sizes.append(np_buf.size)
            extracted_buf.append(np_buf.reshape(-1,))
        return np.concatenate(extracted_buf)

    def set(self, values):
        used_params = 0
        momentum_buffer_list = []
        for i in range(len(self.sizes)):
            np_buf = values[used_params: used_params + self.sizes[i]].reshape(self.shapes[i])
            buf = torch.Tensor(np_buf).cuda()
            momentum_buffer_list.append(buf)
        curr_id = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.optimizer.state[p]['momentum_buffer'] = momentum_buffer_list[curr_id].clone()
                    curr_id += 1
async def main(cfg):
    best_prec1 = 0
    torch.manual_seed(239)

    print('Consensus agent: {}'.format(cfg.agent_token))
    consensus_specific = ConsensusSpecific(cfg)
    consensus_specific.init_consensus()

    # Check the save_dir exists or not
    cfg.save_dir = os.path.join(cfg.save_dir, str(cfg.agent_token))
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[cfg.arch]())
    model.cuda()
    print('{}: Created model'.format(cfg.agent_token))

    statistics = ModelStatistics(cfg.agent_token)

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            if cfg.logging:
                print("=> loading checkpoint '{}'".format(cfg.resume))
            checkpoint = torch.load(cfg.resume)
            cfg.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            if 'statistics' in checkpoint.keys():
                statistics = pickle.loads(checkpoint['statistics'])
            elif os.path.isfile(os.path.join(cfg.resume, 'statistics.pickle')):
                statistics = ModelStatistics.load_from_file(os.path.join(cfg.resume, 'statistics.pickle'))
            model.load_state_dict(checkpoint['state_dict'])
            if cfg.logging:
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(cfg.evaluate, checkpoint['epoch']))
        else:
            if cfg.logging:
                print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    print('{}: Loading dataset...'.format(cfg.agent_token))
    train_loader = get_agent_train_loader(cfg.agent_token, cfg.batch_size)
    print('{}: loaded {} batches for train'.format(cfg.agent_token, len(train_loader)))
    val_loader = None if cfg.no_validation else get_agent_val_loader(cfg.agent_token)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if cfg.half:
        model.half()
        criterion.half()

    optimizer = torch.optim.SGD(model.parameters(), cfg.lr,
                                momentum=cfg.momentum,
                                weight_decay=cfg.weight_decay)

    def lr_schedule(epoch):
        if cfg.use_lsr and epoch < cfg.warmup:
            factor = np.power(cfg.total_agents, epoch/cfg.warmup)
        else:
            factor = cfg.total_agents if cfg.use_lsr else 1.0
        if epoch >= 81:
            factor /= 10
        if epoch >= 122:
            factor /= 10
        return factor

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)

    if cfg.arch != 'resnet20':
        print('This code was not intended to be used on resnets other than resnet20')

    if cfg.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.lr * 0.1

    if cfg.evaluate:
        validate(cfg, val_loader, model, criterion)
        return

    await consensus_specific.agent.send_telemetry(TelemetryAgentGeneralInfo(batches_per_epoch=len(train_loader)))

    for epoch in range(cfg.start_epoch, cfg.epochs):
        # train for one epoch
        if cfg.logging:
            print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        statistics.add('train_begin_timestamp', time.time())
        await train(consensus_specific, train_loader, model, criterion, optimizer, epoch, statistics)
        lr_scheduler.step()
        statistics.add('train_end_timestamp', time.time())

        # evaluate on validation set
        statistics.add('validate_begin_timestamp', time.time())
        prec1 = validate(cfg, val_loader, model, criterion)
        statistics.add('validate_end_timestamp', time.time())
        statistics.add('val_precision', prec1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % cfg.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'statistics': pickle.dumps(statistics)
            }, is_best, filename=os.path.join(cfg.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(cfg.save_dir, 'model.th'))
        statistics.dump_to_file(os.path.join(cfg.save_dir, 'statistics.pickle'))

    consensus_specific.stop_consensus()


async def train(consensus_specific, train_loader, model, criterion, optimizer, epoch, statistics):
    """
        Run one train epoch
    """
    cfg = consensus_specific.cfg

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        consensus_specific.batch_counter += 1
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if cfg.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # average model
        if cfg.momentum_consensus:
            await consensus_specific.run_averaging(model, optimizer)
        else:
            await consensus_specific.run_averaging(model)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        await consensus_specific.agent.send_telemetry(TelemetryModelParameters(
                                                          consensus_specific.batch_counter,
                                                          consensus_specific.dump_params(model)
                                                      ))

        if i % cfg.print_freq == 0:
            if cfg.logging:
                print('\rEpoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1), end='')
    if cfg.logging:
        print('\nEpoch took {:.2f} s.'.format(end - start))
    statistics.add('train_precision', top1.avg)
    statistics.add('train_loss', losses.avg)


def validate(cfg, val_loader, model, criterion):
    if cfg.no_validation or val_loader is None:
        return -1.0
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if cfg.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.print_freq == 0:
                if cfg.logging:
                    print('\rTest: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        top1=top1), end='')

    if cfg.logging:
        print('\n * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    cfg = make_config_parser().parse_args()
    asyncio.get_event_loop().run_until_complete(main(cfg))
