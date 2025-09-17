import numpy as np

class LeftAgent:
    """
    A scripted agent that performs a 'stop-and-go' patrol pattern.
    It moves in short bursts and then pauses, creating erratic movement.
    """
    def __init__(self):
        self.step_counter = 0
        self.sequence_index = 0

        # --- Define the Action Palette ---
        right = np.array([1, 0, 0], dtype=np.int32)
        up = np.array([0, 1, 1], dtype=np.int32)
        left = np.array([2, 0, 2], dtype=np.int32)
        down = np.array([0, 2, 0], dtype=np.int32)
        idle = np.array([0, 0, 0], dtype=np.int32) # The "pause" action

        # --- Customize the Pattern Here ---
        # The sequence of actions the agent will perform.
        self.action_sequence = [
            right,
            idle,
            down,
            idle,
            left,
            idle,
            up,
            idle
        ]

        # The duration (in steps) for each corresponding action in the sequence.
        self.duration_sequence = [
            15,  # Move right for 15 steps
            20,  # Pause for 20 steps
            15,  # Move down for 15 steps
            20,  # Pause for 20 steps
            15,  # Move left for 15 steps
            20,  # Pause for 20 steps
            15,  # Move up for 15 steps
            20   # Pause for 20 steps
        ]

    def act(self, observation):
        """
        Returns the next action in the stop-and-go sequence.
        The 'observation' is ignored, as this is a pre-scripted pattern.
        """
        # Get the current action from the sequence
        action = self.action_sequence[self.sequence_index]

        # Check if it's time to move to the next action in the sequence
        current_duration = self.duration_sequence[self.sequence_index]
        if self.step_counter >= current_duration:
            # Move to the next action and reset the step counter
            self.sequence_index = (self.sequence_index + 1) % len(self.action_sequence)
            self.step_counter = 0

        self.step_counter += 1
        return action

    def reset(self):
        """Resets the sequence back to the beginning."""
        self.step_counter = 0
        self.sequence_index = 0
