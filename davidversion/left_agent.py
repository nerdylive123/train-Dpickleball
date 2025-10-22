# left_agent.py
import numpy as np
import os
import torch
import cv2
from sb3_contrib import RecurrentPPO


def _stack_to_uint8_strip(arr: np.ndarray) -> np.ndarray:
    """
    Convert a (C,H,W) array (float32 or any) to a single grayscale strip image (H, W*C) uint8.
    - If values look like [0,1], scale by 255.
    - Otherwise, min-max normalize per whole array to [0,255].
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    assert arr.ndim == 3, f"Expected (C,H,W), got {arr.shape}"
    v = arr.astype(np.float32)

    vmin = float(np.nanmin(v))
    vmax = float(np.nanmax(v))
    if vmax <= 1.5 and vmin >= -0.5:
        # Likely [0,1] (Unity outputs), clamp and scale
        v = np.clip(v, 0.0, 1.0) * 255.0
    else:
        # Robust normalize to [0,255]
        rng = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        v = (v - vmin) / rng * 255.0
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Tile channels horizontally
    c, h, w = v.shape
    strip = np.concatenate([v[i] for i in range(c)], axis=1)  # (H, W*C)
    return strip


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
        return [0, 1, np.random.randint(0,3, dtype=np.int32)]  # just keep moving right until the middle of the field

    def pattern_wait_stop_then_right(self):
        self.__step += 1
        if self.__step < 35:
            return np.array([0, 0, 0], dtype=np.int32)  # wait for 50 steps
        else:
            return np.array([0, 1, 0], dtype=np.int32)  # then move right

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
        return self.pattern_wait_stop_then_right()

    def reset(self):
        self.time_elapsed = 0
        self.__step = 0


class ModelLeftAgent:
    def __init__(self, model_path: str, device: str | None = None, deterministic: bool = True,
                 debug: bool = False,
                 debug_dir: str = "davidversion/debug_frames/left",
                 debug_first_n: int = 50,
                 debug_every_k: int = 10000):
        self.model_path = model_path if os.path.exists(model_path) else f"{model_path}.zip"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model for left agent not found at {self.model_path}")
        full_agent = RecurrentPPO.load(self.model_path, device="cpu")
        self.model = full_agent.policy
        self.state = None
        self.deterministic = deterministic
        # One-time logging flags per episode
        self._warned_shape_once = False
        self._logged_adjust_once = False
        # Expected input channels from the trained policy's observation space
        self.expected_stack = getattr(self.model.observation_space, 'shape', (1,))[0]
        print(f"LeftAgent loaded model from {self.model_path}")

        # Debug config
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_first_n = debug_first_n
        self.debug_every_k = debug_every_k
        self._dbg_step = 0
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[ModelLeftAgent] Debug enabled; saving to {self.debug_dir}")

    def _maybe_save(self, obs_chw: np.ndarray, tag: str):
        if not self.debug:
            return
        s = self._dbg_step
        if s < self.debug_first_n or (self.debug_every_k > 0 and s % self.debug_every_k == 0):
            try:
                strip = _stack_to_uint8_strip(obs_chw)
                path = os.path.join(self.debug_dir, f"{tag}_{s:06d}.png")
                cv2.imwrite(path, strip)
                if s < 3:
                    print(f"[ModelLeftAgent] Saved {tag} -> {path} shape={obs_chw.shape}")
            except Exception as e:
                if s < 3:
                    print(f"[ModelLeftAgent] Failed saving {tag}: {e}")

    def act(self, observation):
        if observation is None:
            raise ValueError("ModelLeftAgent requires observation input")
        obs = observation

        # Save BEFORE adjustments (as received from mylib)
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            if not self._warned_shape_once:
                print(f"[ModelLeftAgent] Received obs shape={obs.shape} "
                      f"min={float(np.min(obs)):.3f} max={float(np.max(obs)):.3f}")
                self._warned_shape_once = True
            self._maybe_save(obs, "before")

        # observation may be single-frame (1,H,W); pad/repeat to expected stack if needed
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            c, h, w = obs.shape
            if c == 1 and self.expected_stack > 1:
                obs = np.repeat(obs, self.expected_stack, axis=0)

        # Save AFTER adjustments (still CHW)
        if isinstance(obs, np.ndarray) and obs.ndim == 3:
            if not self._logged_adjust_once:
                print(f"[ModelLeftAgent] Using obs for policy shape={obs.shape} "
                      f"min={float(np.min(obs)):.3f} max={float(np.max(obs)):.3f}")
                self._logged_adjust_once = True
            self._maybe_save(obs, "after")

        # add batch dimension
        obs = np.expand_dims(obs, axis=0)
        action, self.state = self.model.predict(obs, state=self.state, episode_start=None, deterministic=self.deterministic)
        if self.debug and self._dbg_step < 5:
            print(f"LeftAgent action: {action[0]}")
        self._dbg_step += 1
        return action[0]

    def reset(self):
        self.state = None
        self._dbg_step = 0
        self._warned_shape_once = False
        self._logged_adjust_once = False


class RobustModelLeftAgent:

    def __init__(self, model_path: str, device: str | None = None, deterministic: bool = True,
                 debug: bool = False,
                 debug_dir: str = "davidversion/debug_frames/left",
                 debug_first_n: int = 50,
                 debug_every_k: int = 10000):
        import os
        import numpy as np
        from sb3_contrib import RecurrentPPO

        self.model_path = model_path if os.path.exists(model_path) else f"{model_path}.zip"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model for left agent not found at {self.model_path}")

        full_agent = RecurrentPPO.load(self.model_path, device=device or "cpu")
        self.policy = full_agent.policy
        self.policy.set_training_mode(False)

        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)
        self.deterministic = deterministic
        # One-time logging flags
        self._warned_shape_once = False
        self._logged_adjust_once = False

        # Debug config
        self.debug = debug
        self.debug_dir = debug_dir
        self.debug_first_n = debug_first_n
        self.debug_every_k = debug_every_k
        self._dbg_step = 0
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            print(f"[RobustModelLeftAgent] Debug enabled; saving to {self.debug_dir}")

        # Get expected observation shape from policy
        obs_space = self.policy.observation_space
        self.expected_shape = obs_space.shape
        print(f"[RobustModelLeftAgent] Loaded model from {self.model_path}")
        print(f"[RobustModelLeftAgent] Expected observation shape: {self.expected_shape}")

    def _maybe_save(self, obs_chw: np.ndarray, tag: str):
        if not self.debug:
            return
        s = self._dbg_step
        if s < self.debug_first_n or (self.debug_every_k > 0 and s % self.debug_every_k == 0):
            try:
                strip = _stack_to_uint8_strip(obs_chw)
                path = os.path.join(self.debug_dir, f"{tag}_{s:06d}.png")
                cv2.imwrite(path, strip)
                if s < 3:
                    print(f"[RobustModelLeftAgent] Saved {tag} -> {path} shape={obs_chw.shape}")
            except Exception as e:
                if s < 3:
                    print(f"[RobustModelLeftAgent] Failed saving {tag}: {e}")

    def act(self, observation):
        import numpy as np

        if observation is None:
            raise ValueError("RobustModelLeftAgent requires observation input")

        obs = np.array(observation, dtype=np.float32)

        # Save BEFORE adjustments
        if obs.ndim == 3:
            if not self._warned_shape_once:
                print(f"[RobustModelLeftAgent] Received obs shape={obs.shape} "
                      f"min={float(np.min(obs)):.3f} max={float(np.max(obs)):.3f}")
                self._warned_shape_once = True
            self._maybe_save(obs, "before")

        # Validate and fix observation shape if needed
        if obs.shape != self.expected_shape:
            # Handle different cases
            if len(obs.shape) == 3:
                c, h, w = obs.shape
                expected_c, expected_h, expected_w = self.expected_shape

                # If frame stack mismatch
                if c != expected_c:
                    if c > expected_c:
                        # Take most recent frames
                        obs = obs[-expected_c:, :, :]
                    else:
                        # Pad with last frame
                        padding = np.repeat(obs[-1:, :, :], expected_c - c, axis=0)
                        obs = np.concatenate([obs, padding], axis=0)

                # If spatial dimensions mismatch, resize to expected (H, W)
                if h != expected_h or w != expected_w:
                    # (C, H, W) -> (H, W, C) for cv2
                    obs_hw_c = np.transpose(obs, (1, 2, 0))
                    # Resize to (expected_w, expected_h)
                    resized = cv2.resize(obs_hw_c, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
                    # Back to (C, H, W) as float32
                    obs = np.transpose(resized.astype(np.float32), (2, 0, 1))

        # Save AFTER adjustments
        if obs.ndim == 3:
            if not self._logged_adjust_once:
                print(f"[RobustModelLeftAgent] Using obs shape {obs.shape} for policy "
                      f"min={float(np.min(obs)):.3f} max={float(np.max(obs)):.3f}")
                self._logged_adjust_once = True
            self._maybe_save(obs, "after")

        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        obs = np.expand_dims(obs, axis=0)

        with np.errstate(all='ignore'):
            action, self.lstm_states = self.policy.predict(
                obs,
                state=self.lstm_states,
                episode_start=self.episode_start,
                deterministic=self.deterministic
            )

        self.episode_start[0] = False

        self._dbg_step += 1
        return action[0]

    def reset(self):
        import numpy as np
        self.lstm_states = None
        self.episode_start = np.ones((1,), dtype=bool)
        self._warned_shape_once = False
        self._logged_adjust_once = False
        self._dbg_step = 0
