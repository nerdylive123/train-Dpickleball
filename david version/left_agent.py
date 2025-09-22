import numpy as np

class LeftAgent:
    def __init__(self):
        self.time_elapsed = 0

    def act(self, observation):
        # Always return a valid action for MultiDiscrete([3 3 3])
        return np.array([1, 1, 1])

    def reset(self):
        self.time_elapsed = 0
