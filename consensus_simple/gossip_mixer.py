import numpy as np
from copy import deepcopy
from consensus_simple.mixer import Mixer


class GossipMixer(Mixer):
    def __init__(self, topology, logger):
        super().__init__(topology, logger)

    def mix(self, params_dict, times=1, neighbors_num=1):
        names = list(params_dict.keys())
        names_for_gossip = np.random.choice(names, size=times)

        result_params_dict = deepcopy(params_dict)

        for name in names_for_gossip:
            result_params_dict[name] = self._mix_params_once(params_dict, name, neighbors_num=neighbors_num)

        return result_params_dict

    def _mix_params_once(self, params_dict, agent, neighbors_num=1):
        neighbors = list(set(self.topology[agent].keys()) - {agent})
        neighbors_for_gossip = list(np.random.choice(neighbors, size=neighbors_num)) + [agent]
        return sum(1. / len(neighbors_for_gossip) * params_dict[neighbor_name]
                   for neighbor_name in neighbors_for_gossip)
