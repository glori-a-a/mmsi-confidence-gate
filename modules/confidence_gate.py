from dataclasses import dataclass
import torch
import torch.nn.functional as F
import numpy as np

class MultiClassConfidenceGate:
    """
    Multi-class confidence-gated EMA smoothing.
    - Maintain state m_t over C classes.
    - Update only when top1-top2 margin >= tau.
    """
    def __init__(self, num_classes, alpha=0.2, tau=0.12):
        self.C = num_classes
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.reset()

    def reset(self):
        self.m = np.ones(self.C, dtype=np.float32) / self.C

    def step(self, p_vec):
        p = np.asarray(p_vec, dtype=np.float32)
        # confidence = margin between top1 and top2
        top2 = np.partition(p, -2)[-2:]
        conf = float(top2[-1] - top2[-2])

        if conf >= self.tau:
            self.m = (1.0 - self.alpha) * self.m + self.alpha * p
        # else: keep m

        y_hat = int(np.argmax(self.m))
        return y_hat, conf, self.m.copy()

@dataclass
class GateConfig:
    alpha: float = 0.2          
    tau: float = 0.15           # confidence threshold
    use_hysteresis: bool = False
    margin_on: float = 0.20     # optional: for hysteresis on
    margin_off: float = 0.10    # optional: for hysteresis off

class CategoricalStateSmoother:
    """
    Confidence-gated EMA smoothing for multi-class predictions.

    Given logits_t (B,C), compute probs p_t.
    confidence c_t = p_max - p_second (margin).
    If c_t >= tau: m_t = (1-alpha) m_{t-1} + alpha p_t
    else:          m_t = m_{t-1}

    Output: smoothed probs m_t and smoothed prediction argmax(m_t).
    """
    def __init__(self, num_classes: int, cfg: GateConfig, device=None):
        self.C = num_classes
        self.cfg = cfg
        self.device = device
        self.reset()

    def reset(self):
        self.m = None  

    @torch.no_grad()
    def step(self, logits: torch.Tensor):
        """
        logits: (B,C)
        returns:
          m: (B,C) smoothed probs
          pred: (B,) argmax(m)
          conf: (B,) margin confidence
          updated: (B,) bool
        """
        probs = F.softmax(logits, dim=-1)  # (B,C)
        B, C = probs.shape
        assert C == self.C

        # margin confidence: top1 - top2
        top2 = torch.topk(probs, k=2, dim=-1).values  
        conf = top2[:, 0] - top2[:, 1]              

        if self.m is None:
            # init state as uniform (or first probs). Uniform is safer if you want "no bias".
            self.m = torch.full_like(probs, 1.0 / C)
            if self.device is not None:
                self.m = self.m.to(self.device)

        gate = (conf >= self.cfg.tau).float().unsqueeze(-1)  

        # EMA update only when gate=1
        self.m = (1.0 - self.cfg.alpha * gate) * self.m + (self.cfg.alpha * gate) * probs

        pred = torch.argmax(self.m, dim=-1)  
        updated = (gate.squeeze(-1) > 0.0)
        return self.m, pred, conf, updated
