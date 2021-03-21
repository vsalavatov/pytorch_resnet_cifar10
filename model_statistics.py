import pickle


class ModelStatistics:
    def __init__(self, token, save_path=None):
        self.token = token
        self.data = {}
        self.save_path = save_path

    def add(self, key, val):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(val)

    def crop(self, key):
        return self.data[key]

    def dump_to_file(self, path=None):
        if path is None:
            path = self.save_path

        if path is None:
            raise ValueError('No path specified. Use argument path=')

        with open(path, 'wb') as f:
            f.write(pickle.dumps(self))

    @staticmethod
    def load_from_file(path):
        with open(path, 'rb') as f:
            return pickle.loads(f.read())
