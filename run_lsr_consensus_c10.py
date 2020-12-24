import argparse
import asyncio
import sys
import os

sys.path.append('./distributed-learning/')
from utils.consensus_tcp import ConsensusMaster

import consensus_trainer

parser = argparse.ArgumentParser()
parser.add_argument('--master-host', default='127.0.0.1', type=str)
parser.add_argument('--master-port', default=8999, type=int)
parser.add_argument('--agent-start-port', default=11000, type=int)
parser.add_argument('--debug', dest='debug', action='store_true')

async def run(args):
    topology = [(i, (i + 1) % 10) for i in range(10)]
    args = parser.parse_args()

    master = ConsensusMaster(topology, '127.0.0.1', args.master_port, debug=True if args.debug else False)
    master_task = asyncio.create_task(master.serve_forever())
    await asyncio.sleep(1)  # wait until master initialize

    agent_tasks = []
    for token in range(10):
        cfg = consensus_trainer.make_config_parser()
        agent_args = cfg.parse_args([
                                  '--agent-token', f'{token}',
                                  '--agent-host', f'{args.master_host}',
                                  '--agent-port', f'{args.agent_start_port + token}',
                                  '--master-host', f'{args.master_host}',
                                  '--master-port', f'{args.master_port}',
                                  '--total-agents', '10',
                                  '--use-prepared-data',
                                  '--use-lsr',
                                  '--save-dir', os.environ['CHECKPOINT_PATH']
                              ]
                              + (['--init-leader', '--enable-log'] if token == 0 else [])
                              + (['--debug-consensus'] if args.debug else [])
                              + ([] if (token == 0 or token == 5) else ['--no-validation'])
                              )
        agent_tasks.append(consensus_trainer.main(agent_args))
    await asyncio.wait(agent_tasks)
    master_task.cancel()

if __name__ == '__main__':
    args = parser.parse_args()
    asyncio.run(run(args))
