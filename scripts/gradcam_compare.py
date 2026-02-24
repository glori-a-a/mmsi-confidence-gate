import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_json", required=True)
    p.add_argument("--candidate_json", required=True)
    p.add_argument("--baseline_name", default="baseline")
    p.add_argument("--candidate_name", default="candidate")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def load_cam_json(path):
    payload = json.loads(Path(path).read_text())
    samples = payload.get("samples", [])
    cams = []
    correct = []
    top1prob = []
    for s in samples:
        cam = np.asarray(s.get("cam_tokens", []), dtype=np.float64)
        if cam.size == 0:
            continue
        cams.append(cam)
        correct.append(int(s.get("pred_label", -1) == s.get("gt_label", -2)))
        probs = s.get("probs", [])
        top1prob.append(float(max(probs)) if probs else float("nan"))
    if not cams:
        raise ValueError(f"No CAM samples found in {path}")
    cams = np.stack(cams, axis=0)
    correct = np.asarray(correct, dtype=np.int64)
    top1prob = np.asarray(top1prob, dtype=np.float64)
    return {
        "payload": payload,
        "cams": cams,
        "correct": correct,
        "top1prob": top1prob,
    }


def safe_entropy(p):
    p = np.clip(p, 1e-9, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def compute_stats(bundle):
    cams = bundle["cams"]
    cams = cams / (cams.sum(axis=1, keepdims=True) + 1e-9)
    ent = safe_entropy(cams)
    peak = cams.max(axis=1)
    top1_idx = cams.argmax(axis=1)
    top3_mass = np.sort(cams, axis=1)[:, -3:].sum(axis=1)
    correct = bundle["correct"]

    def _mean_std(x):
        return float(np.mean(x)), float(np.std(x))

    stats = {
        "num_samples": int(cams.shape[0]),
        "num_tokens": int(cams.shape[1]),
        "mean_cam": cams.mean(axis=0).tolist(),
        "entropy_mean": _mean_std(ent)[0],
        "entropy_std": _mean_std(ent)[1],
        "peak_mass_mean": _mean_std(peak)[0],
        "peak_mass_std": _mean_std(peak)[1],
        "top3_mass_mean": _mean_std(top3_mass)[0],
        "top3_mass_std": _mean_std(top3_mass)[1],
        "pred_conf_mean": float(np.nanmean(bundle["top1prob"])),
        "correct_acc_from_export": float(correct.mean()) if correct.size > 0 else float("nan"),
        "token_hist_argmax": np.bincount(top1_idx, minlength=cams.shape[1]).astype(int).tolist(),
    }

    if correct.size > 0 and correct.sum() > 0 and correct.sum() < len(correct):
        for flag, key in [(1, "correct"), (0, "wrong")]:
            idx = correct == flag
            stats[f"{key}_n"] = int(idx.sum())
            stats[f"{key}_entropy_mean"] = float(ent[idx].mean())
            stats[f"{key}_peak_mass_mean"] = float(peak[idx].mean())
            stats[f"{key}_top3_mass_mean"] = float(top3_mass[idx].mean())
    return stats


def _polyline(points, color, width=2):
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />'


def build_svg_line_chart(mean_a, mean_b, name_a, name_b, out_path):
    w, h = 900, 360
    ml, mr, mt, mb = 60, 20, 30, 50
    plot_w = w - ml - mr
    plot_h = h - mt - mb
    n = len(mean_a)
    ymax = max(max(mean_a), max(mean_b), 1e-6)

    def map_pt(i, v):
        x = ml + (plot_w * i / max(1, n - 1))
        y = mt + plot_h * (1.0 - v / ymax)
        return x, y

    pts_a = [map_pt(i, v) for i, v in enumerate(mean_a)]
    pts_b = [map_pt(i, v) for i, v in enumerate(mean_b)]

    grid = []
    for gy in range(5):
        y = mt + plot_h * gy / 4
        val = ymax * (1 - gy / 4)
        grid.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{w-mr}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        grid.append(f'<text x="8" y="{y+4:.1f}" font-size="12" fill="#555">{val:.3f}</text>')
    for i in range(n):
        x = ml + (plot_w * i / max(1, n - 1))
        grid.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{h-mb}" stroke="#f2f2f2" stroke-width="1"/>')
        grid.append(f'<text x="{x:.1f}" y="{h-20}" text-anchor="middle" font-size="12" fill="#555">T{i}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
<rect width="100%" height="100%" fill="white"/>
<text x="{w/2:.1f}" y="20" text-anchor="middle" font-size="16" font-family="sans-serif">Mean Grad-CAM Token Distribution ({name_a} vs {name_b})</text>
{''.join(grid)}
<rect x="{ml}" y="{mt}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#999" stroke-width="1"/>
{_polyline(pts_a, "#1f77b4", 2.5)}
{_polyline(pts_b, "#d62728", 2.5)}
<circle cx="{ml+20}" cy="{h-35}" r="4" fill="#1f77b4"/><text x="{ml+32}" y="{h-30}" font-size="13" font-family="sans-serif">{name_a}</text>
<circle cx="{ml+180}" cy="{h-35}" r="4" fill="#d62728"/><text x="{ml+192}" y="{h-30}" font-size="13" font-family="sans-serif">{name_b}</text>
</svg>
"""
    Path(out_path).write_text(svg)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a = load_cam_json(args.baseline_json)
    b = load_cam_json(args.candidate_json)
    sa = compute_stats(a)
    sb = compute_stats(b)

    mean_a = np.asarray(sa["mean_cam"], dtype=np.float64)
    mean_b = np.asarray(sb["mean_cam"], dtype=np.float64)
    build_svg_line_chart(mean_a, mean_b, args.baseline_name, args.candidate_name, out_dir / "mean_cam_compare.svg")

    comparison = {
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "baseline": sa,
        "candidate": sb,
        "delta_candidate_minus_baseline": {
            "acc_from_export": sb["correct_acc_from_export"] - sa["correct_acc_from_export"],
            "entropy_mean": sb["entropy_mean"] - sa["entropy_mean"],
            "peak_mass_mean": sb["peak_mass_mean"] - sa["peak_mass_mean"],
            "top3_mass_mean": sb["top3_mass_mean"] - sa["top3_mass_mean"],
            "pred_conf_mean": sb["pred_conf_mean"] - sa["pred_conf_mean"],
        },
    }
    (out_dir / "gradcam_comparison.json").write_text(json.dumps(comparison, indent=2))

    md = []
    md.append("# Grad-CAM Comparison")
    md.append("")
    md.append(f"- Baseline: `{args.baseline_name}`")
    md.append(f"- Candidate: `{args.candidate_name}`")
    md.append("")
    md.append("| Metric | Baseline | Candidate | Delta (cand - base) |")
    md.append("|---|---:|---:|---:|")
    for k in ["correct_acc_from_export", "pred_conf_mean", "entropy_mean", "peak_mass_mean", "top3_mass_mean"]:
        md.append(
            f"| {k} | {sa[k]:.4f} | {sb[k]:.4f} | {sb[k]-sa[k]:+.4f} |"
        )
    md.append("")
    md.append("Notes:")
    md.append("- Lower `entropy_mean` suggests more concentrated CAM over tokens.")
    md.append("- Higher `peak_mass_mean` / `top3_mass_mean` suggests stronger focus on fewer tokens.")
    md.append("- `correct_acc_from_export` is computed from exported predictions and should align with test accuracy on the exported set.")
    md.append("")
    md.append("See `mean_cam_compare.svg` for token-wise mean CAM distribution.")
    (out_dir / "gradcam_comparison.md").write_text("\n".join(md) + "\n")

    print(f"[OK] Saved comparison JSON: {out_dir / 'gradcam_comparison.json'}")
    print(f"[OK] Saved comparison MD:   {out_dir / 'gradcam_comparison.md'}")
    print(f"[OK] Saved SVG figure:      {out_dir / 'mean_cam_compare.svg'}")


if __name__ == "__main__":
    main()
