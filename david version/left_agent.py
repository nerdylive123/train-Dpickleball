import numpy as np

class LeftAgent:
    def __init__(self):
        self.time_elapsed = 0
        self.__step = 0

    def pattern_1(self):
        # Cycle between left and right movements each 5 time steps
        if (self.__step // 12) % 2 == 0:
            action = np.array([0, 2, 2])  # Move left
            self.__step += 2
        else:
            action = np.array([0, 1, 0])
            self.__step += 1
        return action

    def pattern_simple_right(self):
        self.__step += 1
        return [0, 1, 0] #just keep moving right until the middle of the field

    def pattern_simple2(self):
        # move left [0,2,0] for 10 steps then next should be stop [0,0,0] for 60 steps then
        # move right [0,1,0] for 10 steps then stop [0,0,0] for 50 steps
        cycle_length = 100

        step_in_cycle = self.__step % cycle_length
        self.__step += 1

        if step_in_cycle < 10:
            action = np.array([0, 2, 0])
        elif step_in_cycle < 70:
            action = np.array([0, 0, 0])
        elif step_in_cycle < 80:
            action = np.array([0, 1, 0])
        else:
            action = np.array([0, 0, 0])
        return action

    def pattern_still(self):
        self.__step += 1
        return np.array([0, 0, 0])


    def act(self, observation):
        return self.pattern_simple_right()

    def reset(self):
        self.time_elapsed = 0
        self.__step = 0