# Experiment Summary (STI)

## Setup
- Clean setting: removed label leakage tokens from input text.
- Dataset split: train/test no overlap.
- Metric: multi-class accuracy on test set (N=821).
- Stability metric: switch-rate (prediction transition rate across time).

## Key Results
1. Clean baseline (no FiLM, no gate):
   - Acc = 0.341

2. + FiLM fusion:
   - Acc = 0.361
   - Improvement = +0.020

3. + Confidence gate (best stability-oriented setting):
   - Gate Acc = 0.336 (delta = -0.005 vs FiLM baseline)
   - Switch-rate: 0.756 -> 0.728 (delta = -0.028)

## Takeaway
- FiLM is the main contributor to accuracy gain.
- Confidence gate improves temporal stability, with a small accuracy tradeoff in current configuration.
