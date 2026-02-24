# Grad-CAM Comparison

- Baseline: `baseline`
- Candidate: `v2f2`

| Metric | Baseline | Candidate | Delta (cand - base) |
|---|---:|---:|---:|
| correct_acc_from_export | 0.3625 | 0.4875 | +0.1250 |
| pred_conf_mean | 0.4939 | 0.9596 | +0.4658 |
| entropy_mean | 1.2591 | 1.3720 | +0.1129 |
| peak_mass_mean | 0.3551 | 0.2895 | -0.0656 |
| top3_mass_mean | 0.5247 | 0.5731 | +0.0484 |

Notes:
- Lower `entropy_mean` suggests more concentrated CAM over tokens.
- Higher `peak_mass_mean` / `top3_mass_mean` suggests stronger focus on fewer tokens.
- `correct_acc_from_export` is computed from exported predictions and should align with test accuracy on the exported set.

See `mean_cam_compare.svg` for token-wise mean CAM distribution.
