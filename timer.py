from time import time
class Timer(object):
    def __init__(self, description):
        self.description = description
        print(f"{self.description}...")
    def __enter__(self):
        self.start = time()
    def __exit__(self, type, value, traceback):
        self.end = time()
        print(f"Done ({self.end - self.start} seconds).")