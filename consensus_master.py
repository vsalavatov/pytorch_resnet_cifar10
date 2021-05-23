import os, sys
import argparse
import asyncio
from typing import Any
from dataclasses import dataclass
import numpy as np

from model_statistics import ModelStatistics

sys.path.append('./distributed-learning/')
from utils.consensus_tcp import ConsensusMaster
from utils.consensus_tcp import TelemetryProcessor


@dataclass
class TelemetryModelParameters:
    batch_number: int
    parameters: Any


@dataclass
class TelemetryAgentGeneralInfo:
    telemetries_per_epoch: int


class ResNet20TelemetryProcessor(TelemetryProcessor):
    def __init__(self, destination_path, topology, resume=False):
        self.destination_path = destination_path
        self.agents = list(set([uv[0] for uv in topology] + [uv[1] for uv in topology]))

        self.stats = ModelStatistics.load_from_file(destination_path) if resume \
            else ModelStatistics('MASTER TELEMETRY', save_path=destination_path)

        self.agent_params_by_iter = dict()
        self.agent_general_info = dict()

    def process(self, token, payload):
        if isinstance(payload, TelemetryModelParameters):
            if payload.batch_number not in self.agent_params_by_iter.keys():
                self.agent_params_by_iter[payload.batch_number] = dict()
            self.agent_params_by_iter[payload.batch_number][token] = payload.parameters
            if len(self.agent_params_by_iter[payload.batch_number]) == len(self.agents):
                params = self.agent_params_by_iter[payload.batch_number]
                avg_params = np.mean([params[agent] for agent in self.agents], axis=0)
                deviation_params = {agent: params[agent] - avg_params for agent in self.agents}
                self.stats.add('param_deviation_L1',
                               {agent: np.linalg.norm(deviation_params[agent], ord=1) for agent in self.agents})
                self.stats.add('param_deviation_L2',
                               {agent: np.linalg.norm(deviation_params[agent], ord=2) for agent in self.agents})
                self.stats.add('param_deviation_Linf',
                               {agent: np.linalg.norm(deviation_params[agent], ord=np.inf) for agent in self.agents})

                arr_params = np.array([params[agent] for agent in self.agents])
                cvar = np.std(arr_params, axis=0) / np.abs(np.mean(arr_params, axis=0))
                mean_cvar = np.mean(cvar)
                percentiles = (25, 50, 75, 80, 90, 95, 99, 99.9)
                cvar_pctls = np.percentile(cvar, percentiles)

                self.stats.add('abs_coef_of_var_percentiles', list(zip(percentiles, cvar_pctls)))
                self.stats.add('abs_coef_of_var_mean', mean_cvar)

                self.stats.dump_to_file()
                del self.agent_params_by_iter[payload.batch_number]
        elif isinstance(payload, TelemetryAgentGeneralInfo):
            self.agent_general_info[token] = payload
            if len(self.agent_general_info) == len(self.agents):
                self.stats.add('telemetries_per_epoch',
                               {agent: self.agent_general_info[agent].telemetries_per_epoch for agent in self.agents})
                self.stats.dump_to_file()
        else:
            raise ValueError(f'Got unsupported payload from {token}: {payload!r}')


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
