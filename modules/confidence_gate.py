from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class TTMConfig:
    alpha: float = 0.2      # EMA update speed
    tau: float = 0.15       # confidence threshold
    theta_on: float = 0.65  # hysteresis high threshold
    theta_off: float = 0.35 # hysteresis low threshold
    init_m: float = 0.5     # initial state
    init_y: int = 0         # initial decision 0/1

    def __post_init__(self):
        assert 0.0 <= self.init_m <= 1.0
        assert 0.0 <= self.theta_off < self.theta_on <= 1.0
        assert 0.0 <= self.tau <= 0.5
        assert 0.0 < self.alpha <= 1.0


class TTMStateSmoother:
    """
    Confidence-gated state update + hysteresis decision.
    Input: per-frame probabilities p_t in [0,1].
    Output: smoothed binary decisions y_hat_t in {0,1}, plus state m_t.
    """

    def __init__(self, cfg: Optional[TTMConfig] = None):
        self.cfg = cfg or TTMConfig()
        self.reset()

    def reset(self):
        self.m = float(self.cfg.init_m)
        self.y = int(self.cfg.init_y)

    @staticmethod
    def confidence(p: float) -> float:
        return abs(p - 0.5)

    def step(self, p: float) -> int:
        # clamp p
        p = float(np.clip(p, 0.0, 1.0))
        c = self.confidence(p)

        # gated EMA update
        if c >= self.cfg.tau:
            self.m = (1.0 - self.cfg.alpha) * self.m + self.cfg.alpha * p
        # else: keep m unchanged

        # hysteresis decision
        if self.m >= self.cfg.theta_on:
            self.y = 1
        elif self.m <= self.cfg.theta_off:
            self.y = 0
        # else: keep y unchanged

        return self.y

    def run(self, probs: Iterable[float], reset: bool = True) -> dict:
        if reset:
            self.reset()

        ys: List[int] = []
        ms: List[float] = []
        cs: List[float] = []

        for p in probs:
            p = float(np.clip(p, 0.0, 1.0))
            cs.append(self.confidence(p))
            ys.append(self.step(p))
            ms.append(self.m)

        return {
            "y_hat": np.array(ys, dtype=np.int64),
            "m": np.array(ms, dtype=np.float32),
            "c": np.array(cs, dtype=np.float32),
        }
