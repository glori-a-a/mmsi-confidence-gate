import argparse
import os
import json
import random
from functools import partial
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
from modules.confidence_gate import GateConfig, MultiClassConfidenceGate


seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer(language_model):
    if language_model == "bert":
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained("bert-base-uncased")
    if language_model == "roberta":
        from transformers import RobertaTokenizer
        return RobertaTokenizer.from_pretrained("roberta-base")
    if language_model == "electra":
        from transformers import ElectraTokenizer
        return ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    raise ValueError(f"Unsupported language model: {language_model}")


def to_jsonable(x):
    """Recursively convert numpy / torch objects into JSON-serializable python types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def ensure_dir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def sanity_required_paths(args):
    required = {
        "txt_dir": args.txt_dir,
        "txt_labeled_dir": args.txt_labeled_dir,
        "keypoint_dir": args.keypoint_dir,
        "meta_dir": args.meta_dir,
        "data_split_file": args.data_split_file,
        "checkpoint_file": args.checkpoint_file,
    }
    for k, v in required.items():
        if v in ["enter_the_path", "..."]:
            raise FileNotFoundError(
                f"Argument --{k} is still placeholder ('{v}'). Please pass the real path."
            )


def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--model_name", type=str, default="model_name")
    parser.add_argument("--task", type=str, default="STI", choices=["STI", "PCR", "MPP"])
    parser.add_argument("--language_model", type=str, default="bert", choices=["bert", "roberta", "electra"])
    parser.add_argument("--use_film_fusion", action="store_true",
                        help="Enable FiLM-conditioned visual fusion (must match training).")
    parser.add_argument("--max_people_num", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)

    # data paths
    parser.add_argument("--txt_dir", type=str, required=True)
    parser.add_argument("--txt_labeled_dir", type=str, required=True)
    parser.add_argument("--keypoint_dir", type=str, required=True)
    parser.add_argument("--meta_dir", type=str, required=True)
    parser.add_argument("--data_split_file", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, required=True)

    # smoother params
    parser.add_argument("--use_ttm_smoother", action="store_true",
                        help="Apply confidence-gated smoother on binary target stream.")
    parser.add_argument("--use_confidence_gate", action="store_true",
                        help="Alias of --use_ttm_smoother.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.12)
    parser.add_argument("--theta_on", type=float, default=0.65)
    parser.add_argument("--theta_off", type=float, default=0.35)
    parser.add_argument("--use_multiclass_gate", action="store_true",
                        help="Apply confidence gate on full multi-class probability streams.")
    parser.add_argument("--grid_search_gate", action="store_true",
                        help="Grid-search alpha/tau for multi-class gate.")
    parser.add_argument("--grid_alphas", type=str, default="0.05,0.1,0.2,0.3",
                        help="Comma-separated alpha values for gate grid search.")
    parser.add_argument("--grid_taus", type=str, default="0.05,0.1,0.15,0.2,0.25",
                        help="Comma-separated tau values for gate grid search.")
    parser.add_argument("--gate_pred_policy", type=str, default="smoothed",
                        choices=["smoothed", "raw_on_confident", "blend_on_uncertain", "safe_gate", "safe_flip"],
                        help="How to convert gated probabilities to final prediction.")
    parser.add_argument("--gate_output_tau", type=float, default=-1.0,
                        help="Threshold for raw_on_confident policy. If <0, use --tau.")
    parser.add_argument("--gate_gap_reset_sec", type=int, default=-1,
                        help="If >0, reset gate state when time gap exceeds this value.")
    parser.add_argument("--grid_gate_policies", type=str, default="smoothed,safe_flip,safe_gate,raw_on_confident",
                        help="Comma-separated prediction policies for gate grid search.")
    parser.add_argument("--grid_gap_resets", type=str, default="-1,10,20",
                        help="Comma-separated gap-reset seconds for gate grid search.")
    parser.add_argument("--gate_blend_weight", type=float, default=0.5,
                        help="Blend weight for blend_on_uncertain policy.")
    parser.add_argument("--gate_blend_tau", type=float, default=-1.0,
                        help="Confidence threshold for blend_on_uncertain. If <0, use --tau.")
    parser.add_argument("--grid_blend_weights", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated blend weights for blend_on_uncertain search.")
    parser.add_argument("--grid_blend_taus", type=str, default="0.05,0.1,0.15,0.2",
                        help="Comma-separated blend taus for blend_on_uncertain search.")
    parser.add_argument("--grid_rank_mode", type=str, default="acc",
                        choices=["acc", "acc_minus_switch"],
                        help="How to rank grid-search candidates.")
    parser.add_argument("--grid_switch_penalty", type=float, default=0.05,
                        help="Penalty weight for switch-rate when rank_mode=acc_minus_switch.")

    # multi-class -> binary by choosing K
    parser.add_argument("--binary_target_label", type=int, default=-1,
                        help="If >=0, convert multi-class logits to prob stream p_t=P(y==K) and gt_t=(label==K). "
                             "This enables the binary smoother on STI(6-class).")

    # saving outputs
    parser.add_argument("--save_path", type=str, default="",
                        help="If set, save results JSON here (e.g., runs/sti_test/results.json).")

    # demo smoother only
    parser.add_argument("--demo_smoother", action="store_true",
                        help="Run a tiny synthetic demo of the confidence gate and save JSON to --save_path.")

    return parser.parse_args()


def run_smoother_demo(cfg: GateConfig):
    gate = MultiClassConfidenceGate(num_classes=2, alpha=cfg.alpha, tau=cfg.tau)

    p = np.concatenate([
        np.full(20, 0.1, dtype=np.float32),
        np.linspace(0.1, 0.9, 20, dtype=np.float32),
        (0.5 + 0.08 * np.random.randn(40)).astype(np.float32),
        np.full(20, 0.9, dtype=np.float32),
    ])
    p = np.clip(p, 0.0, 1.0)

    y_hat, conf, m = [], [], []
    for pk in p.tolist():
        yh, cf, state = gate.step([1.0 - pk, pk])
        y_hat.append(int(yh))
        conf.append(float(cf))
        m.append(state.tolist())

    out = {
        "y_hat": y_hat,
        "conf": conf,
        "m": m,
    }
    return {
        "cfg": {"alpha": cfg.alpha, "tau": cfg.tau, "use_hysteresis": cfg.use_hysteresis},
        "p": p,
        "out": out,
    }


def parse_float_list(csv_text: str):
    vals = []
    for s in csv_text.split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError(f"Empty float list from: '{csv_text}'")
    return vals


def parse_str_list(csv_text: str):
    vals = [s.strip() for s in csv_text.split(",") if s.strip()]
    if not vals:
        raise ValueError(f"Empty string list from: '{csv_text}'")
    return vals


def parse_int_list(csv_text: str):
    vals = []
    for s in csv_text.split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError(f"Empty int list from: '{csv_text}'")
    return vals


@torch.no_grad()
def eval_multiclass_baseline(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for _, _, tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        mask_idxs = mask_idxs.to(device, non_blocking=True)
        keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
        speaker_labels = speaker_labels.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)

        logits = model(tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=False)
        pred = torch.argmax(logits, dim=1)

        total += task_labels.size(0)
        correct += (pred == task_labels).sum().item()

    return correct / max(total, 1), int(total)


@torch.no_grad()
def eval_binary_with_optional_smoother(model, loader, device, target_k: int,
                                      use_smoother: bool, cfg: GateConfig):
    model.eval()

   
    streams = defaultdict(list)

    for file_names, time_secs, tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        mask_idxs = mask_idxs.to(device, non_blocking=True)
        keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
        speaker_labels = speaker_labels.to(device, non_blocking=True)

        logits = model(tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=False)
        probs = torch.softmax(logits, dim=1)  

        prob_k = probs[:, target_k].detach().cpu().numpy()  
        gt = (task_labels.numpy() == target_k).astype(np.int64) 

        for fn, ts, pk, g in zip(list(file_names), list(time_secs), prob_k.tolist(), gt.tolist()):
            streams[fn].append((int(ts), float(pk), int(g)))

    
    base_correct = 0
    base_total = 0

    smooth_correct = 0
    smooth_total = 0

    per_file = {}

    for fn, items in streams.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        t = [x[0] for x in items_sorted]
        p = np.array([x[1] for x in items_sorted], dtype=np.float32)
        y = np.array([x[2] for x in items_sorted], dtype=np.int64)

        y_base = (p >= 0.5).astype(np.int64)

        base_correct += int((y_base == y).sum())
        base_total += int(len(y))

        if use_smoother:
            gate = MultiClassConfidenceGate(num_classes=2, alpha=cfg.alpha, tau=cfg.tau)
            y_smooth = []
            conf_stream = []
            m_target = []

            for pk in p.tolist():
                y_hat, conf, m = gate.step([1.0 - pk, pk])
                y_smooth.append(int(y_hat))
                conf_stream.append(float(conf))
                m_target.append(float(m[1]))
            y_smooth = np.array(y_smooth, dtype=np.int64)

            smooth_correct += int((y_smooth == y).sum())
            smooth_total += int(len(y))

            per_file[fn] = {
                "num": int(len(y)),
                "time_sec": t,
                "p_target": p,
                "gt": y,
                "pred_base": y_base,
                "pred_smooth": y_smooth,
                "conf_margin": np.array(conf_stream, dtype=np.float32),
                "p_target_smooth": np.array(m_target, dtype=np.float32),
            }
        else:
            per_file[fn] = {
                "num": int(len(y)),
                "time_sec": t,
                "p_target": p,
                "gt": y,
                "pred_base": y_base,
            }

    base_acc = base_correct / max(base_total, 1)
    smooth_acc = None
    if use_smoother:
        smooth_acc = smooth_correct / max(smooth_total, 1)

    stats = {
        "num_files": int(len(streams)),
        "num_samples": int(base_total),
        "target_label": int(target_k),
        "baseline_binary_acc": float(base_acc),
        "smoother_enabled": bool(use_smoother),
        "smoother_binary_acc": float(smooth_acc) if smooth_acc is not None else None,
    }

    return stats, per_file


@torch.no_grad()
def collect_multiclass_streams(model, loader, device):
    model.eval()
    streams = defaultdict(list)
    for file_names, time_secs, tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        mask_idxs = mask_idxs.to(device, non_blocking=True)
        keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
        speaker_labels = speaker_labels.to(device, non_blocking=True)

        logits = model(tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=False)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        gt = task_labels.numpy().astype(np.int64)

        for fn, ts, pvec, y in zip(list(file_names), list(time_secs), probs, gt):
            streams[fn].append((int(ts), pvec.astype(np.float32), int(y)))

    for fn in list(streams.keys()):
        streams[fn] = sorted(streams[fn], key=lambda x: x[0])
    return streams


def eval_multiclass_with_gate_from_streams(streams, cfg: GateConfig,
                                           pred_policy: str = "smoothed",
                                           output_tau: float = None,
                                           gap_reset_sec: int = -1,
                                           blend_weight: float = 0.5,
                                           blend_tau: float = None):
    base_correct, gate_correct, total = 0, 0, 0
    base_switches, gate_switches, total_transitions = 0, 0, 0
    per_file = {}
    eff_output_tau = cfg.tau if (output_tau is None or output_tau < 0) else float(output_tau)
    eff_blend_tau = cfg.tau if (blend_tau is None or blend_tau < 0) else float(blend_tau)

    for fn, items in streams.items():
        gate = None
        if cfg is not None:
            c = len(items[0][1])
            gate = MultiClassConfidenceGate(num_classes=c, alpha=cfg.alpha, tau=cfg.tau)
        prev_ts = None

        t = []
        y = []
        base_pred = []
        gate_pred = []

        for ts, pvec, gt in items:
            if gate is not None and gap_reset_sec > 0 and prev_ts is not None:
                if int(ts) - int(prev_ts) > int(gap_reset_sec):
                    gate.reset()
            prev_ts = ts

            t.append(int(ts))
            y.append(int(gt))

            pb = int(np.argmax(pvec))
            base_pred.append(pb)
            base_correct += int(pb == gt)
            raw_top2 = np.partition(pvec, -2)[-2:]
            raw_conf = float(raw_top2[-1] - raw_top2[-2])

            if gate is not None:
                pg_raw, conf, mvec = gate.step(pvec)
                if pred_policy == "raw_on_confident" and conf >= eff_output_tau:
                    pg = pb
                elif pred_policy == "safe_gate":
                    # Conservative policy: use gate only when gate state is
                    # more decisive than raw probabilities, otherwise fallback.
                    mtop2 = np.partition(mvec, -2)[-2:]
                    gate_conf = float(mtop2[-1] - mtop2[-2])
                    pg = int(pg_raw) if gate_conf > raw_conf else pb
                elif pred_policy == "safe_flip":
                    # Flip class only when raw is uncertain and gate is much more decisive.
                    mtop2 = np.partition(mvec, -2)[-2:]
                    gate_conf = float(mtop2[-1] - mtop2[-2])
                    same = int(pg_raw) == pb
                    if same:
                        pg = pb
                    elif raw_conf < eff_output_tau and gate_conf > (raw_conf + 0.05):
                        pg = int(pg_raw)
                    else:
                        pg = pb
                elif pred_policy == "blend_on_uncertain" and conf < eff_blend_tau:
                    mixed = (1.0 - float(blend_weight)) * pvec + float(blend_weight) * mvec
                    pg = int(np.argmax(mixed))
                else:
                    pg = int(pg_raw)
            else:
                pg = pb
            gate_pred.append(int(pg))
            gate_correct += int(pg == gt)

            total += 1

        per_file[fn] = {
            "num": len(items),
            "time_sec": np.array(t, dtype=np.int64),
            "gt": np.array(y, dtype=np.int64),
            "pred_base": np.array(base_pred, dtype=np.int64),
            "pred_gate": np.array(gate_pred, dtype=np.int64),
        }
        if len(base_pred) >= 2:
            base_switches += int(np.sum(np.array(base_pred[1:]) != np.array(base_pred[:-1])))
            gate_switches += int(np.sum(np.array(gate_pred[1:]) != np.array(gate_pred[:-1])))
            total_transitions += int(len(base_pred) - 1)

    return {
        "num_files": int(len(streams)),
        "num_samples": int(total),
        "baseline_multiclass_acc": float(base_correct / max(total, 1)),
        "gate_multiclass_acc": float(gate_correct / max(total, 1)),
        "delta_acc": float((gate_correct - base_correct) / max(total, 1)),
        "baseline_switch_rate": float(base_switches / max(total_transitions, 1)),
        "gate_switch_rate": float(gate_switches / max(total_transitions, 1)),
        "delta_switch_rate": float((gate_switches - base_switches) / max(total_transitions, 1)),
    }, per_file


def grid_search_multiclass_gate(streams, alphas, taus, policies, gap_resets,
                                blend_weights, blend_taus, rank_mode="acc",
                                switch_penalty=0.05):
    best = None
    all_results = []
    for a in alphas:
        for t in taus:
            for pol in policies:
                for gr in gap_resets:
                    bws = blend_weights if pol == "blend_on_uncertain" else [0.0]
                    bts = blend_taus if pol == "blend_on_uncertain" else [float(t)]
                    for bw in bws:
                        for bt in bts:
                            cfg = GateConfig(alpha=float(a), tau=float(t))
                            stats, _ = eval_multiclass_with_gate_from_streams(
                                streams=streams,
                                cfg=cfg,
                                pred_policy=pol,
                                output_tau=float(t),
                                gap_reset_sec=int(gr),
                                blend_weight=float(bw),
                                blend_tau=float(bt),
                            )
                            row = {
                                "alpha": float(a),
                                "tau": float(t),
                                "pred_policy": str(pol),
                                "gap_reset_sec": int(gr),
                                "blend_weight": float(bw),
                                "blend_tau": float(bt),
                                "gate_multiclass_acc": float(stats["gate_multiclass_acc"]),
                                "baseline_multiclass_acc": float(stats["baseline_multiclass_acc"]),
                                "delta_acc": float(stats["delta_acc"]),
                                "baseline_switch_rate": float(stats["baseline_switch_rate"]),
                                "gate_switch_rate": float(stats["gate_switch_rate"]),
                                "delta_switch_rate": float(stats["delta_switch_rate"]),
                            }
                            if rank_mode == "acc_minus_switch":
                                row["rank_score"] = (
                                    row["gate_multiclass_acc"] - float(switch_penalty) * row["gate_switch_rate"]
                                )
                            else:
                                row["rank_score"] = row["gate_multiclass_acc"]
                            all_results.append(row)
                            if best is None or row["rank_score"] > best["rank_score"]:
                                best = row

    all_results = sorted(all_results, key=lambda x: x["rank_score"], reverse=True)
    return best, all_results

def main():
    args = parse_args()
    if args.use_confidence_gate:
        args.use_ttm_smoother = True

    cfg = GateConfig(alpha=args.alpha, tau=args.tau)

    # demo mode
    if args.demo_smoother:
        if not args.save_path:
            raise ValueError("--demo_smoother requires --save_path")
        payload = run_smoother_demo(cfg)
        ensure_dir_for_file(args.save_path)
        with open(args.save_path, "w") as f:
            json.dump(to_jsonable(payload), f, indent=2)
        print(f"[OK] Smoother demo saved to: {args.save_path}")
        return

    # normal evaluation needs paths + checkpoint
    sanity_required_paths(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalBaseline(
        args.max_people_num,
        args.language_model,
        use_film_fusion=args.use_film_fusion
    ).to(device)

    ckpt = torch.load(args.checkpoint_file, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    print(f"[OK] Loaded checkpoint: {args.checkpoint_file}")

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer

    test_dataset = SocialDataset(args, is_training=False)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, tokenizer),
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # 1) multi-class baseline
    acc_mc, n_mc = eval_multiclass_baseline(model, test_loader, device)
    print()
    print(f"[Multi-class baseline] acc={acc_mc:.3f} (N={n_mc})")

    multiclass_gate_stats = None
    multiclass_grid_best = None
    multiclass_grid_all = None
    multiclass_per_file = None
    if args.use_multiclass_gate or args.grid_search_gate:
        streams = collect_multiclass_streams(model, test_loader, device)

        if args.grid_search_gate:
            alphas = parse_float_list(args.grid_alphas)
            taus = parse_float_list(args.grid_taus)
            policies = parse_str_list(args.grid_gate_policies)
            gap_resets = parse_int_list(args.grid_gap_resets)
            blend_weights = parse_float_list(args.grid_blend_weights)
            blend_taus = parse_float_list(args.grid_blend_taus)
            multiclass_grid_best, multiclass_grid_all = grid_search_multiclass_gate(
                streams, alphas, taus, policies, gap_resets, blend_weights, blend_taus,
                rank_mode=args.grid_rank_mode, switch_penalty=args.grid_switch_penalty
            )
            print()
            print(f"[Grid search] best alpha={multiclass_grid_best['alpha']}, tau={multiclass_grid_best['tau']}, "
                  f"policy={multiclass_grid_best['pred_policy']}, gap_reset={multiclass_grid_best['gap_reset_sec']}, "
                  f"blend_w={multiclass_grid_best['blend_weight']}, blend_tau={multiclass_grid_best['blend_tau']}")
            print(f"[Grid search] best gate_acc={multiclass_grid_best['gate_multiclass_acc']:.3f}"
                  f" (delta={multiclass_grid_best['delta_acc']:+.3f})")
            print(f"[Grid search] switch_rate base={multiclass_grid_best['baseline_switch_rate']:.3f} "
                  f"gate={multiclass_grid_best['gate_switch_rate']:.3f} "
                  f"(delta={multiclass_grid_best['delta_switch_rate']:+.3f}) "
                  f"| rank_mode={args.grid_rank_mode} score={multiclass_grid_best['rank_score']:.4f}")

        if args.use_multiclass_gate:
            output_tau = args.gate_output_tau if args.gate_output_tau >= 0 else args.tau
            blend_tau = args.gate_blend_tau if args.gate_blend_tau >= 0 else args.tau
            multiclass_gate_stats, multiclass_per_file = eval_multiclass_with_gate_from_streams(
                streams=streams,
                cfg=cfg,
                pred_policy=args.gate_pred_policy,
                output_tau=output_tau,
                gap_reset_sec=args.gate_gap_reset_sec,
                blend_weight=args.gate_blend_weight,
                blend_tau=blend_tau,
            )
            print()
            print(f"[Multi-class gate] baseline_acc={multiclass_gate_stats['baseline_multiclass_acc']:.3f}")
            print(f"[Multi-class gate] gate_acc={multiclass_gate_stats['gate_multiclass_acc']:.3f}")
            print(f"[Multi-class gate] switch_rate base={multiclass_gate_stats['baseline_switch_rate']:.3f} "
                  f"gate={multiclass_gate_stats['gate_switch_rate']:.3f} "
                  f"(delta={multiclass_gate_stats['delta_switch_rate']:+.3f})")
            print(f"[Multi-class gate] delta={multiclass_gate_stats['delta_acc']:+.3f}"
                  f" (alpha={args.alpha}, tau={args.tau}, policy={args.gate_pred_policy}, "
                  f"gap_reset={args.gate_gap_reset_sec}, blend_w={args.gate_blend_weight}, "
                  f"blend_tau={blend_tau})")

    # 2) optional binary + smoother
    binary_stats = None
    per_file = None
    if args.use_ttm_smoother:
        if args.binary_target_label < 0:
            print("[WARN] --use_ttm_smoother was set but --binary_target_label < 0.")
            print("       STI is multi-class. To apply the binary smoother, pick a target label K:")
            print("       e.g., --binary_target_label 0  (meaning: 'is the addressee Player0?')")
        else:
            binary_stats, per_file = eval_binary_with_optional_smoother(
                model=model,
                loader=test_loader,
                device=device,
                target_k=args.binary_target_label,
                use_smoother=True,
                cfg=cfg
            )
            print()
            print(f"[Binary target K={args.binary_target_label}] baseline_acc={binary_stats['baseline_binary_acc']:.3f}")
            print(f"[Binary target K={args.binary_target_label}] smoother_acc={binary_stats['smoother_binary_acc']:.3f}")
            print(f"Smoother cfg: alpha={args.alpha}, tau={args.tau}")
            print("[NOTE] theta_on/theta_off are ignored by modules/confidence_gate.py.")

            if "unknown" in per_file:
                print()
                print("[NOTE] file_name/time_sec meta not found in dataset (all grouped under 'unknown').")
                print("       For REAL temporal smoothing, add dataset.data_meta in dataloader.py (2 lines).")

    print(f"Device: {device}")
    print(f"Model: {args.model_name}")

    # 3) optional save
    if args.save_path:
        payload = {
            "model_name": args.model_name,
            "task": args.task,
            "checkpoint_file": args.checkpoint_file,
            "device": str(device),
            "multiclass_baseline": {
                "acc": float(acc_mc),
                "num_samples": int(n_mc),
            },
            "binary_eval": binary_stats,
            "multiclass_gate_eval": multiclass_gate_stats,
            "multiclass_gate_grid_best": multiclass_grid_best,
            "multiclass_gate_grid_all": multiclass_grid_all,
            "smoother_cfg": {
                "alpha": float(args.alpha),
                "tau": float(args.tau),
                "gate_pred_policy": str(args.gate_pred_policy),
                "gate_output_tau": float(args.gate_output_tau),
                "gate_gap_reset_sec": int(args.gate_gap_reset_sec),
                "gate_blend_weight": float(args.gate_blend_weight),
                "gate_blend_tau": float(args.gate_blend_tau),
                "grid_rank_mode": str(args.grid_rank_mode),
                "grid_switch_penalty": float(args.grid_switch_penalty),
                "theta_on": float(args.theta_on),
                "theta_off": float(args.theta_off),
                "note": "Binary eval only exists when --use_ttm_smoother and --binary_target_label>=0. "
                        "theta_on/theta_off are kept only for CLI compatibility.",
            },
        }

        if per_file is not None:
            payload["per_file_streams"] = per_file
        if multiclass_per_file is not None:
            payload["per_file_multiclass_gate"] = multiclass_per_file

        ensure_dir_for_file(args.save_path)
        with open(args.save_path, "w") as f:
            json.dump(to_jsonable(payload), f, indent=2)

        print(f"[OK] Saved results to: {args.save_path}")


if __name__ == "__main__":
    main()
