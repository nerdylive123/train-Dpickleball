import numpy as np

class LeftAgent:
    def __init__(self):
        self.time_elapsed = 0

    def act(self, observation):
        # Cycle between left and right movements each 1 time steps
        if self.time_elapsed % 2 == 0:
            action = np.array([0, 1, 0])
        else:
            action = np.array([0, 2, 0])
        self.time_elapsed += 1
        return action

    def reset(self):
        self.time_elapsed = 0
