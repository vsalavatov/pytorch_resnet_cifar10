from copy import copy
from consensus_simple.mixer import Mixer


def _normalize_weights(weights):
    w_sum = sum(weights.values())
    for key in weights:
        weights[key] = weights[key] / w_sum
    return weights


class WeightedMixer(Mixer):
    def __init__(self, topology, logger, weights, dev_metric=None):
        super().__init__(topology, logger, dev_metric=dev_metric)
        self.weights = _normalize_weights(copy(weights))

    def mix(self, agents_params):
        return self._mix_params_once(agents_params)

    def _mix_params_once(self, agents_params):
        # [agent_name]: params
        result_params = {}
        new_weights = copy(self.weights)
        for agent_name in agents_params:
            new_params = sum(self.weights[neighbor]*agents_params[neighbor] for neighbor in self.topology[agent_name])
            new_weights[agent_name] = sum(self.weights[neighbor] for neighbor in self.topology[agent_name])
            new_params = new_params / new_weights[agent_name]
            result_params[agent_name] = new_params
        self.weights = _normalize_weights(new_weights)
        return result_params
