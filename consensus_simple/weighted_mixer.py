from copy import deepcopy
from consensus_simple.mixer import Mixer


def _normalize_weights(weights):
    w_sum = sum(weights.values())
    for key in weights:
        weights[key] = weights[key] / w_sum
    return weights


class WeightedMixer(Mixer):
    def __init__(self, topology, logger, weights, lr_schedule, lr, dev_metric=None):
        super().__init__(topology, logger, dev_metric=dev_metric)
        self.weights = _normalize_weights(deepcopy(weights))
        self.lr_schedule = lr_schedule
        self.lr = lr

    def _calc_lr(self, iteration):
        return self.lr * self.lr_schedule(iteration)

    def mix(self, agents_params, iteration):
        lr = self._calc_lr(iteration)
        return self._mix_params_once(agents_params, lr)

    def _mix_params_once(self, agents_params, lr):
        # [agent_name]: params
        result_params = {}
        for agent_name in agents_params:
            new_params = deepcopy(agents_params[agent_name])
            diff_sum = lr * sum(agents_params[neighbor] - agents_params[agent_name]
                                for neighbor in self.topology[agent_name]
                                if neighbor != agent_name)
            new_params = new_params + diff_sum / self.weights[agent_name]
            result_params[agent_name] = new_params
        return result_params
