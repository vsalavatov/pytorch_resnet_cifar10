from model_statistics import ModelStatistics


class StatisticCollector(ModelStatistics):
    def __init__(self, token, logger, save_path=None):
        super().__init__(token, save_path)
        self.logger = logger

    def add(self, key, val):
        super().add(key, val)
        self.logger.info('StatisticCollector {} add {} with value {}'.format(self.token, key, val))
        return self
