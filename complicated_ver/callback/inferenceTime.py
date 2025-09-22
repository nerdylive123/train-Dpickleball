from abc import ABC
from collections import deque
import numpy as np
import time
import torch
from stable_baselines3.common.callbacks import BaseCallback

class InferenceTimerCallback(BaseCallback, ABC):
    def __init__(self, target_ms=10, window=2000, verbose=1, per_env=True):
        super().__init__(verbose)
        self.target_ms = target_ms
        self.window = window
        self.per_env = per_env
        self.times_ms = deque(maxlen=window)
        self.times_raw_ms = deque(maxlen=window)
        self.batch_sizes = deque(maxlen=window)
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

            # Derive batch size from inputs if available
            bs = 1
            try:
                if isinstance(inputs, tuple) and len(inputs) > 0 and hasattr(inputs[0], 'shape'):
                    bs = int(inputs[0].shape[0])
            except Exception:
                bs = 1

            self.batch_sizes.append(bs)
            self.times_raw_ms.append(dt_ms)
            # Normalize by batch for per-env latency
            self.times_ms.append(dt_ms / max(1, bs))

        # Hook the top-level policy module once to get end-to-end forward time
        self._handles.append(policy.register_forward_pre_hook(pre_hook))
        self._handles.append(policy.register_forward_hook(post_hook))

    def _on_rollout_end(self) -> None:
        if len(self.times_ms) == 0:
            return
        arr = np.array(self.times_ms, dtype=np.float64)
        arr_raw = np.array(self.times_raw_ms, dtype=np.float64)

        mean = arr.mean()
        p95 = np.percentile(arr, 95)
        p99 = np.percentile(arr, 99)

        mean_raw = arr_raw.mean()
        p99_raw = np.percentile(arr_raw, 99)

        if self.verbose:
            print(f"[Inference] per-env ms -> mean: {mean:.2f} | p95: {p95:.2f} | p99: {p99:.2f} | n={len(arr)}")
            print(f"            batch  ms -> mean: {mean_raw:.2f} | p99: {p99_raw:.2f} (avg bs≈{np.mean(self.batch_sizes):.1f})")
        if self.target_ms is not None and p99 > self.target_ms:
            raise RuntimeError(f"Inference per-env p99 {p99:.2f} ms exceeds target {self.target_ms:.2f} ms")

    def _on_training_end(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _on_step(self) -> bool:
        # This callback does not act on individual steps; required by BaseCallback
        return True