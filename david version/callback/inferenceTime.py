from abc import ABC
from collections import deque
import numpy as np
import time
import torch
from stable_baselines3.common.callbacks import BaseCallback

class InferenceTimerCallback(BaseCallback, ABC):
    def __init__(self, target_ms=None, window=2000, verbose=1):
        super().__init__(verbose)
        self.target_ms = target_ms
        self.window = window
        self.times_ms = deque(maxlen=window)
        self._handles = []

    def _on_training_start(self) -> None:
        policy = self.model.policy

        def pre_hook(module, inputs):
            # Only time inference (rollouts), not training
            if module.training:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            module._t0 = time.perf_counter()

        def post_hook(module, inputs, output):
            if module.training:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - module._t0) * 1000.0
            self.times_ms.append(dt_ms)

        # Hook the top-level policy module once to get end-to-end forward time
        self._handles.append(policy.register_forward_pre_hook(pre_hook))
        self._handles.append(policy.register_forward_hook(post_hook))

    def _on_rollout_end(self) -> None:
        if len(self.times_ms) == 0:
            return
        arr = np.array(self.times_ms, dtype=np.float64)
        mean = arr.mean()
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)
        if self.verbose:
            print(f"[Inference] mean: {mean:.2f} ms | p95: {p95:.2f} ms | p99: {p99:.2f} ms | n={len(arr)}")
        if self.target_ms is not None and p99 > self.target_ms:
            raise RuntimeError(f"Inference p99 {p99:.2f} ms exceeds target {self.target_ms:.2f} ms")

    def _on_training_end(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _on_step(self) -> bool:
        # This callback does not act on individual steps; required by BaseCallback
        return True