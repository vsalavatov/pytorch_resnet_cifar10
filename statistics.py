class Statistics:
    def __init__(self, token):
        self.token = token
        self.data = {}
        self.epoch = None
        self._first_epoch = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        if self._first_epoch is None:
            self._first_epoch = epoch

    def add(self, key, val):
        if self.epoch not in self.data.keys():
            self.data[self.epoch] = {}
        self.data[self.epoch][key] = val

    def crop(self, key):
        if self.epoch is None:
            return None
        return [self.data.get(i, {}).get(key, None)
                for i in range(self._first_epoch, self.epoch + 1)]
