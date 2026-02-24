import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_csv", required=True, help="Path to sweep summary.csv")
    p.add_argument("--output_md", required=True, help="Output markdown report path")
    p.add_argument("--topk", type=int, default=20, help="Show top-K rows by accuracy")
    return p.parse_args()


def safe_load_json(path_str):
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def main():
    args = parse_args()
    summary_path = Path(args.summary_csv)
    rows = []
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            acc = None
            try:
                acc = float(r.get("test_acc", ""))
            except ValueError:
                acc = None
            payload = safe_load_json(r.get("eval_json", ""))
            num_samples = None
            if payload and isinstance(payload, dict):
                num_samples = payload.get("multiclass_baseline", {}).get("num_samples")
                mk = payload.get("multiclass_gate_eval")
                if mk:
                    r["gate_acc"] = mk.get("gate_multiclass_acc")
            r["test_acc_float"] = acc
            r["num_samples"] = num_samples
            rows.append(r)

    rows_sorted = sorted(
        rows,
        key=lambda x: (-1e9 if x["test_acc_float"] is None else -x["test_acc_float"], x.get("trial_id", "")),
    )

    topk_rows = rows_sorted[: max(1, args.topk)]

    best = next((r for r in rows_sorted if r["test_acc_float"] is not None), None)

    lines = []
    lines.append("# Ablation Report")
    lines.append("")
    lines.append(f"- Summary CSV: `{summary_path}`")
    lines.append(f"- Total trials: {len(rows)}")
    if best:
        lines.append(
            f"- Best trial: `{best['trial_id']}` | acc={best['test_acc_float']:.4f} "
            f"| V={best.get('vision_film_layers')} F={best.get('fusion_film_layers')} "
            f"| center={best.get('use_center_loss')}"
        )
    lines.append("")
    lines.append("## Top Results")
    lines.append("")
    lines.append("| Rank | Trial | V-FiLM | F-FiLM | Center | Acc | N | Eval JSON |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---|")
    rank = 0
    for r in topk_rows:
        rank += 1
        acc = "" if r["test_acc_float"] is None else f"{r['test_acc_float']:.4f}"
        n = "" if r.get("num_samples") is None else str(r["num_samples"])
        lines.append(
            f"| {rank} | {r.get('trial_id','')} | {r.get('vision_film_layers','')} | "
            f"{r.get('fusion_film_layers','')} | {r.get('use_center_loss','')} | "
            f"{acc} | {n} | `{r.get('eval_json','')}` |"
        )

    lines.append("")
    lines.append("## All Trials (raw)")
    lines.append("")
    lines.append("| Trial | V-FiLM | F-FiLM | Center | Acc | Checkpoint Dir |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r in rows_sorted:
        acc = "" if r["test_acc_float"] is None else f"{r['test_acc_float']:.4f}"
        lines.append(
            f"| {r.get('trial_id','')} | {r.get('vision_film_layers','')} | "
            f"{r.get('fusion_film_layers','')} | {r.get('use_center_loss','')} | "
            f"{acc} | `{r.get('checkpoint_dir','')}` |"
        )

    out_path = Path(args.output_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[OK] Report saved to: {out_path}")


if __name__ == "__main__":
    main()
