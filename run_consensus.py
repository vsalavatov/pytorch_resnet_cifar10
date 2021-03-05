import argparse
import asyncio
import sys
import os

sys.path.append('./distributed-learning/')
from utils.consensus_tcp import ConsensusMaster

import consensus_trainer

parser = argparse.ArgumentParser()
parser.add_argument('--world-size', '-n', type=int,
                    help='You should either specify both world-size and topology'
                         ' or use custom topology using topology-file option')
parser.add_argument('--topology', choices=['mesh', 'star', 'ring', 'torus'], type=str)
parser.add_argument('--topology-file', type=str)
parser.add_argument('--validation-agents', type=str, help='e.g. --validation-agents="0,3,6" or --validation-agents="*"')

parser.add_argument('--consensus-freq', dest='consensus_frequency', type=int, default=1,
                        help='freq>0 -> do averaging <freq> times per batch, '
                             'freq<0 -> do averaging once per (-freq) batches')
parser.add_argument('--use-consensus-rounds', dest='use_consensus_rounds', action='store_true',
                    help='do consensus rounds instead of fixed number of consensus iterations')
parser.add_argument('--consensus-rounds-precision', dest='consensus_rounds_precision', type=float, default=1e-4)
parser.add_argument('--use-lsr', dest='use_lsr', action='store_true')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')

parser.add_argument('--master-host', default='127.0.0.1', type=str)
parser.add_argument('--master-port', default=8999, type=int)
parser.add_argument('--agent-start-port', default=11000, type=int)
parser.add_argument('--debug', dest='debug', action='store_true')


def make_topology(args):
    bad_args_msg = 'You should either specify both world-size and topology'\
                   ' or use custom topology using topology-file option'
    if args.world_size is not None and args.topology is not None:
        n = args.world_size
        if args.topology == 'mesh':
            return [(i, j) for i in range(args.world_size) for j in range(i + 1, n)], n
        elif args.topology == 'star':
            return [(0, j) for j in range(1, n)], n
        elif args.topology == 'ring':
            return [(j, (j + 1) % n) for j in range(n)], n
        elif args.topology == 'torus':
            side = n ** 0.5
            if side * side != n:
                raise ValueError('topology=torus => world size must be exact square')
            return [ (
                       (layer * side + elem),
                       (layer * side + (elem + 1) % side)
                   ) for elem in range(side) for layer in range(side)] \
                   + \
                   [ (
                       (layer * side + elem),
                       ((layer + 1) % side * side + elem)
                   ) for elem in range(side) for layer in range(side)],\
                   n
        else:
            raise ValueError(bad_args_msg)
    if args.topology_file is not None:
        with open(args.topology_file, 'r') as f:
            file_fmt_help = 'File should look like this:\n'\
                            '0 1\n'\
                            '1 2\n'\
                            '2 0\n\n'\
                            'Agent designations must be integers from 0 to n-1 where n is the total number of agents'
            topology = []
            agents = set()
            try:
                for line in f.readlines():
                    tokens = list(map(int, line.strip().split()))
                    if tokens:
                        if len(tokens) != 2:
                            raise ValueError('File is ill-formated')
                        agents.add(tokens[0])
                        agents.add(tokens[1])
                        topology.append((tokens[0], tokens[1]))

                for i in range(len(agents)):
                    if i not in agents:
                        raise ValueError('File is ill-formated')
            except Exception as e:
                print(f'{e} happened while reading the topology file.\n' + file_fmt_help)
                raise e

            return topology, len(agents)
    raise ValueError(bad_args_msg)


def extract_validation_agents(args, total_agents):
    if args.validation_agents is None:
        return []
    if args.validation_agents == '*':
        return list(range(total_agents))
    try:
        return list(map(int, args.validation_agents.strip().split(',')))
    except Exception as e:
        print('validation-agents option should look like this: "0" or "0,3,6" or "*"')
        raise e


async def run(args):
    topology, total_agents = make_topology(args)

    master = ConsensusMaster(topology, '127.0.0.1', args.master_port, debug=True if args.debug else False)
    master_task = asyncio.create_task(master.serve_forever())
    await asyncio.sleep(1)  # wait until master initialize

    validation_agents = extract_validation_agents(args, total_agents)

    agent_tasks = []
    for token in range(total_agents):
        cfg = consensus_trainer.make_config_parser()
        agent_args = cfg.parse_args([
                                  '--agent-token', f'{token}',
                                  '--agent-host', f'{args.master_host}',
                                  '--agent-port', f'{args.agent_start_port + token}',
                                  '--master-host', f'{args.master_host}',
                                  '--master-port', f'{args.master_port}',
                                  '--total-agents', f'{total_agents}',
                                  '--save-dir', os.environ['CHECKPOINT_PATH'],
                                  '--use-prepared-data'
                              ]

                              + (['--consensus-freq', f'{args.consensus_frequency}']
                                 if args.consensus_frequency is not None else [])
                              + (['--use-consensus-rounds'] if args.use_consensus_rounds is not None else [])
                              + (['--consensus-rounds-precision', f'{args.consensus_rounds_precision}']
                                 if args.consensus_rounds_precision is not None else [])
                              + (['--use-lsr'] if args.use_lsr else [])
                              + (['--batch-size', f'{args.batch_size}'] if args.batch_size is not None else [])

                              + (['--init-leader', '--enable-log'] if token == 0 else [])
                              + (['--debug-consensus'] if args.debug else [])
                              + ([] if token in validation_agents else ['--no-validation'])
                              )
        agent_tasks.append(consensus_trainer.main(agent_args))
    await asyncio.wait(agent_tasks)
    master_task.cancel()

if __name__ == '__main__':
    args = parser.parse_args()
    asyncio.run(run(args))
