import numpy as np


def basic_deviation_metric(p1, p2):
    return np.linalg.norm(p1 - p2)


class Mixer(object):
    def __init__(self, topology, logger, dev_metric=None):
        self.topology = topology
        self.logger = logger
        self.dev_metric = basic_deviation_metric
        if dev_metric is not None:
            self.dev_metric = dev_metric

    def mix(self, params_dict, agent):
        return self._mix_params_once(params_dict, agent)

    def _update_stopping_criterion(self, agent_parameters, times_done, max_times, eps):
        return (eps is None or self._get_max_deviation(agent_parameters) < eps) and (times_done >= max_times)

    def _mix_params_once(self, params_dict, agent):
        return sum(params * self.topology[agent][neighbor] for neighbor, params in params_dict.items())

    def _get_max_deviation(self, params):
        max_dev = -1
        for agent1 in params:
            for agent2 in params:
                max_dev = max(max_dev, self.dev_metric(params[agent1], params[agent2]))

        self.logger.debug('Mixer calculate max deviation= {}'.format(max_dev))
        return max_dev
