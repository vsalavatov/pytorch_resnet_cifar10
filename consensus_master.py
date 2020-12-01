import os, sys
import argparse
import asyncio

sys.path.append('./distributed-learning/')
from utils.consensus_tcp import ConsensusMaster

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-host', default='127.0.0.1', type=str)
    parser.add_argument('--master-port', required=True, type=int)
    parser.add_argument('--topology-path', required=True, type=str, help='path to a file describing topology')
    parser.add_argument('--debug', dest='debug', action='store_true')

    args = parser.parse_args()

    topology = []
    with open(args.topology_path, 'r') as f:
        for line in f.readlines():
            tokens = line.strip().split()
            if len(tokens) == 2:
                topology.append(
                    (int(tokens[0]), int(tokens[1]))
                )

    master = ConsensusMaster(topology, args.master_host, args.master_port, debug=True if args.debug else False)

    asyncio.get_event_loop().run_until_complete(master.serve_forever())
