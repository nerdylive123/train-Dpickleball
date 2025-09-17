import numpy as np

class LeftAgent:
    def __init__(self):
        self.time_elapsed = 0

    def act(self, observation):
        # Cycle between left and right movements each 1 time steps
        # if self.time_elapsed % 2 == 0:
        #     action = [0, 1, 0] # right
        # else:
        #     action = [0, 2, 0] #move left
        # self.time_elapsed += 1
        return [0,1,0]

    def reset(self):
        self.time_elapsed = 0
