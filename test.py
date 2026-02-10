import argparse
import os
import json
import random
from functools import partial
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
from modules.confidence_gate import TTMStateSmoother, TTMConfig


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


class SocialDatasetWithMeta(Dataset):
    """
    Wraps SocialDataset and returns:
      (file_name, time_sec, tokens, mask_idx, keypoint_seq, speaker_label, task_label)
    We fetch file_name/time_sec from SocialDataset.data_points which is built in __init__.
    """

    def __init__(self, base_dataset: SocialDataset):
        self.base = base_dataset
        self.has_meta = hasattr(base_dataset, "data_meta")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        tokens, mask_idx, keypoint_seq, speaker_label, task_label = self.base[idx]

        if self.has_meta:           
            meta = self.base.data_meta[idx]
            file_name = meta.get("file_name", "unknown")
            time_sec = int(meta.get("time_sec", idx))
        else:
            file_name = "unknown"
            time_sec = int(idx)

        return file_name, time_sec, tokens, mask_idx, keypoint_seq, speaker_label, task_label


def collate_with_meta(tokenizer, batch):
    """
    batch item:
      (file_name, time_sec, tokens, mask_idx, keypoint_seq, speaker_label, task_label)

    Returns:
      file_names (list[str]),
      time_secs (list[int]),
      tokens (B,L),
      mask_idxs (B,),
      keypoint_seqs (B,6,16,34),
      speaker_labels (B,),
      task_labels (B,)
    """
    file_names, time_secs, tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels = zip(*batch)
    tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels = collate_fn(tokenizer, list(zip(tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels)))
    return list(file_names), list(time_secs), tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels


def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--model_name", type=str, default="model_name")
    parser.add_argument("--task", type=str, default="STI", choices=["STI", "PCR", "MPP"])
    parser.add_argument("--language_model", type=str, default="bert", choices=["bert", "roberta", "electra"])
    parser.add_argument("--max_people_num", type=int, default=6)
    parser.add_argument("--context_length", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)

    # data paths
    parser.add_argument("--txt_dir", type=str, default="enter_the_path")
    parser.add_argument("--txt_labeled_dir", type=str, default="enter_the_path")
    parser.add_argument("--keypoint_dir", type=str, default="enter_the_path")
    parser.add_argument("--meta_dir", type=str, default="enter_the_path")
    parser.add_argument("--data_split_file", type=str, default="enter_the_path")
    parser.add_argument("--checkpoint_file", type=str, default="enter_the_path")

    # smoother params
    parser.add_argument("--use_ttm_smoother", action="store_true",
                        help="Apply confidence-gated hysteresis smoother on a PROBABILITY STREAM.")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.12)
    parser.add_argument("--theta_on", type=float, default=0.65)
    parser.add_argument("--theta_off", type=float, default=0.35)

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


def run_smoother_demo(cfg: TTMConfig):
    smoother = TTMStateSmoother(cfg)

    p = np.concatenate([
        np.full(20, 0.1),
        np.linspace(0.1, 0.9, 20),
        0.5 + 0.08 * np.random.randn(40),
        np.full(20, 0.9),
    ])
    p = np.clip(p, 0.0, 1.0)

    out = smoother.run(p, reset=True)
    return {
        "cfg": {"alpha": cfg.alpha, "tau": cfg.tau, "theta_on": cfg.theta_on, "theta_off": cfg.theta_off},
        "p": p,
        "out": out,
    }


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
                                      use_smoother: bool, cfg: TTMConfig):
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

    smoother = TTMStateSmoother(cfg) if use_smoother else None

    for fn, items in streams.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        t = [x[0] for x in items_sorted]
        p = np.array([x[1] for x in items_sorted], dtype=np.float32)
        y = np.array([x[2] for x in items_sorted], dtype=np.int64)

        y_base = (p >= 0.5).astype(np.int64)

        base_correct += int((y_base == y).sum())
        base_total += int(len(y))

        if smoother is not None:
            out = smoother.run(p, reset=True)  
            y_smooth = np.array(out["y_hat"], dtype=np.int64)

            smooth_correct += int((y_smooth == y).sum())
            smooth_total += int(len(y))

            per_file[fn] = {
                "num": int(len(y)),
                "time_sec": t,
                "p_target": p,
                "gt": y,
                "pred_base": y_base,
                "pred_smooth": y_smooth,
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
    if smoother is not None:
        smooth_acc = smooth_correct / max(smooth_total, 1)

    stats = {
        "num_files": int(len(streams)),
        "num_samples": int(base_total),
        "target_label": int(target_k),
        "baseline_binary_acc": float(base_acc),
        "smoother_enabled": bool(smoother is not None),
        "smoother_binary_acc": float(smooth_acc) if smooth_acc is not None else None,
    }

    return stats, per_file

def main():
    args = parse_args()

    cfg = TTMConfig(alpha=args.alpha, tau=args.tau, theta_on=args.theta_on, theta_off=args.theta_off)

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

    model = MultimodalBaseline(args.max_people_num, args.language_model).to(device)

    ckpt = torch.load(args.checkpoint_file, map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint has no key 'model'. Keys: {list(ckpt.keys())}")
    model.load_state_dict(ckpt["model"])
    print(f"[OK] Loaded checkpoint: {args.checkpoint_file}")

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer

    # base dataset + wrapper with meta
    base_test_dataset = SocialDataset(args, is_training=False)
    test_dataset = SocialDatasetWithMeta(base_test_dataset)

    collate_fn_token = partial(collate_with_meta, tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn_token,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    # 1) multi-class baseline
    acc_mc, n_mc = eval_multiclass_baseline(model, test_loader, device)
    print()
    print(f"[Multi-class baseline] acc={acc_mc:.3f} (N={n_mc})")

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
            print(f"Smoother cfg: alpha={args.alpha}, tau={args.tau}, theta_on={args.theta_on}, theta_off={args.theta_off}")

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
            "smoother_cfg": {
                "alpha": float(args.alpha),
                "tau": float(args.tau),
                "theta_on": float(args.theta_on),
                "theta_off": float(args.theta_off),
                "note": "Binary eval only exists when --use_ttm_smoother and --binary_target_label>=0.",
            },
        }

        if per_file is not None:
            payload["per_file_streams"] = per_file

        ensure_dir_for_file(args.save_path)
        with open(args.save_path, "w") as f:
            json.dump(to_jsonable(payload), f, indent=2)

        print(f"[OK] Saved results to: {args.save_path}")


if __name__ == "__main__":
    main()
