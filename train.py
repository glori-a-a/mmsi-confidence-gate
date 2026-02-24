import argparse
import random
from functools import partial
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
from modules.center_loss import CenterLoss
from utils import AverageMeter, Progbar


seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='model_name')
    parser.add_argument('--task', type=str, default='STI', choices=['STI', 'PCR', 'MPP'])

    parser.add_argument('--txt_dir', type=str, required=True)
    parser.add_argument('--txt_labeled_dir', type=str, required=True)
    parser.add_argument('--keypoint_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)
    parser.add_argument('--data_split_file', type=str, required=True)

    parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints')
    parser.add_argument('--language_model', type=str, default='bert',
                        choices=['bert', 'roberta', 'electra'])
    parser.add_argument('--use_film_fusion', action='store_true',
                        help='Enable FiLM-conditioned visual fusion.')
    parser.add_argument('--film_vision_layers', type=int, default=0,
                        help='Number of FiLM layers on the vision branch. If 0 and --use_film_fusion is set, defaults to 1 for backward compatibility.')
    parser.add_argument('--film_fusion_layers', type=int, default=0,
                        help='Number of FiLM layers on the multimodal fusion branch.')
    parser.add_argument('--film_hidden_dim', type=int, default=512,
                        help='Hidden size of FiLM generators.')
    parser.add_argument('--max_people_num', type=int, default=6)
    parser.add_argument('--context_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--epochs_warmup', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--gate_consistency_lambda', type=float, default=0.0,
                        help='Weight of gate-inspired consistency loss in training (0 disables).')
    parser.add_argument('--gate_consistency_alpha', type=float, default=0.2,
                        help='EMA alpha for training-time gate teacher state.')
    parser.add_argument('--gate_consistency_tau', type=float, default=0.12,
                        help='Confidence margin threshold for gate teacher update.')
    parser.add_argument('--use_center_loss', action='store_true',
                        help='Enable center loss on fused CLS features.')
    parser.add_argument('--center_loss_lambda', type=float, default=0.01,
                        help='Weight for center loss term.')
    parser.add_argument('--center_loss_lr', type=float, default=1e-3,
                        help='Learning rate for center loss centers (separate optimizer param group).')
    parser.add_argument('--enable_gradcam', action='store_true',
                        help='Enable Grad-CAM hooks/cache during forward. Needed for extraction and optional CAM regularization.')
    parser.add_argument('--gradcam_target', type=str, default='vision_post_transformer',
                        choices=['vision_pre_transformer', 'vision_post_transformer', 'fusion_output'],
                        help='Which internal sequence tensor to use for token Grad-CAM.')
    parser.add_argument('--gradcam_loss_lambda', type=float, default=0.0,
                        help='Optional Grad-CAM entropy regularization weight (0 disables).')
    parser.add_argument('--gradcam_loss_mode', type=str, default='entropy',
                        choices=['entropy', 'mass'],
                        help='Grad-CAM regularizer: entropy=minimize entropy; mass=maximize mean activation (implemented as negative mean).')
    parser.add_argument('--gradcam_compute_every', type=int, default=1,
                        help='Compute Grad-CAM every N batches for logging/regularization. 1 means every batch.')
    parser.add_argument('--use_class_weight', action='store_true',
                        help='Use class-weighted cross entropy based on train label distribution.')
    parser.add_argument('--class_weight_mode', type=str, default='sqrt_inv',
                        choices=['inv', 'sqrt_inv'],
                        help='Weight mode: inv=1/freq, sqrt_inv=1/sqrt(freq).')

    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--save_last', action='store_true')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Log training metrics to Weights & Biases.')
    parser.add_argument('--wandb_project', type=str, default='mmsi-fiml-centerloss',
                        help='WandB project name.')
    parser.add_argument('--wandb_entity', type=str, default='',
                        help='WandB entity/team (optional).')
    parser.add_argument('--wandb_name', type=str, default='',
                        help='WandB run name (defaults to model_name).')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='WandB mode.')
    parser.add_argument('--wandb_tags', type=str, default='',
                        help='Comma-separated WandB tags.')

    return parser.parse_args()


class GateTeacherState:
    def __init__(self, num_classes, alpha=0.2, tau=0.12):
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.states = {}

    @torch.no_grad()
    def get_targets_and_update(self, file_names, probs):
        # probs: (B, C), already softmaxed
        top2 = torch.topk(probs, k=2, dim=1).values
        conf = top2[:, 0] - top2[:, 1]
        uncertain_mask = conf < self.tau
        targets = torch.empty_like(probs)

        for i, fn in enumerate(file_names):
            state = self.states.get(fn)
            if state is None:
                state = torch.full(
                    (self.num_classes,), 1.0 / self.num_classes,
                    dtype=probs.dtype, device=probs.device
                )

            targets[i] = state

            if conf[i].item() >= self.tau:
                state = (1.0 - self.alpha) * state + self.alpha * probs[i]

            self.states[fn] = state

        return targets, uncertain_mask


def build_class_weight(train_dataset, num_classes, mode='sqrt_inv'):
    counts = np.zeros(int(num_classes), dtype=np.float64)
    for s in getattr(train_dataset, 'samples', []):
        y = int(s.get('task_label', -1))
        if 0 <= y < num_classes:
            counts[y] += 1.0

    # Avoid divide-by-zero for missing classes.
    counts = np.maximum(counts, 1.0)
    if mode == 'inv':
        w = 1.0 / counts
    else:
        w = 1.0 / np.sqrt(counts)

    # Normalize around 1.0 to keep LR scale stable.
    w = w / np.mean(w)
    return torch.tensor(w, dtype=torch.float32), counts


def get_tokenizer(language_model):
    if language_model == 'bert':
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained('bert-base-uncased')
    elif language_model == 'roberta':
        from transformers import RobertaTokenizer
        return RobertaTokenizer.from_pretrained('roberta-base')
    elif language_model == 'electra':
        from transformers import ElectraTokenizer
        return ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    else:
        raise ValueError(language_model)


def build_model_kwargs(args):
    return {
        'class_num': args.max_people_num,
        'language_model': args.language_model,
        'use_film_fusion': args.use_film_fusion,
        'film_vision_layers': args.film_vision_layers,
        'film_fusion_layers': args.film_fusion_layers,
        'film_hidden_dim': args.film_hidden_dim,
        'enable_gradcam': args.enable_gradcam or args.gradcam_loss_lambda > 0,
    }


def args_for_logging(args):
    payload = {}
    for k, v in vars(args).items():
        if k == 'tokenizer':
            continue
        try:
            json.dumps(v)
            payload[k] = v
        except TypeError:
            payload[k] = str(v)
    return payload


def maybe_init_wandb(args):
    if not args.use_wandb or args.wandb_mode == 'disabled':
        return None
    try:
        import wandb
    except ImportError as e:
        raise ImportError(
            "WandB requested but not installed. Install with `pip install wandb` or disable --use_wandb."
        ) from e

    tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
    init_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_name or args.model_name or None,
        mode=args.wandb_mode,
        config=args_for_logging(args),
        tags=tags or None,
    )
    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:
        # Common cluster case: wandb installed but no API key configured.
        msg = str(e)
        if args.wandb_mode == 'online' and 'No API key configured' in msg:
            print("[WANDB] No API key configured for online mode. Falling back to offline mode.")
            init_kwargs["mode"] = "offline"
            run = wandb.init(**init_kwargs)
        else:
            raise
    return run



def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, use_amp, epoch, args, gate_teacher=None,
                    center_criterion=None):
    model.train()
    meter = AverageMeter()
    meter_ce = AverageMeter()
    meter_gate = AverageMeter()
    meter_center = AverageMeter()
    meter_cam = AverageMeter()
    meter_cam_mean = AverageMeter()
    progbar = Progbar(len(loader.dataset))

    for step_i, (file_names, _, language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels) in enumerate(loader):
        language_tokens = language_tokens.to(device, non_blocking=True)
        mask_idxs = mask_idxs.to(device, non_blocking=True)
        keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
        speaker_labels = speaker_labels.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            model_out = model(
                language_tokens,
                mask_idxs,
                keypoint_seqs,
                speaker_labels,
                warmup=(epoch < args.epochs_warmup),
                return_aux=(center_criterion is not None) or (args.gradcam_loss_lambda > 0)
            )
            if isinstance(model_out, dict):
                outputs = model_out['logits']
                cls_feature = model_out['cls_feature']
            else:
                outputs = model_out
                cls_feature = None

            ce_loss = criterion(outputs, task_labels)
            gate_loss = torch.tensor(0.0, device=device)
            center_loss = torch.tensor(0.0, device=device)
            cam_loss = torch.tensor(0.0, device=device)
            cam_mean = torch.tensor(0.0, device=device)

            if gate_teacher is not None and args.gate_consistency_lambda > 0:
                probs = torch.softmax(outputs, dim=1)
                targets, uncertain_mask = gate_teacher.get_targets_and_update(
                    file_names, probs.detach()
                )
                if uncertain_mask.any():
                    log_probs_u = F.log_softmax(outputs[uncertain_mask], dim=1)
                    targets_u = targets[uncertain_mask]
                    gate_loss = F.kl_div(log_probs_u, targets_u, reduction='batchmean')

            if center_criterion is not None and cls_feature is not None:
                center_loss = center_criterion(cls_feature, task_labels)

            should_compute_cam = (
                (args.gradcam_loss_lambda > 0 or args.enable_gradcam)
                and (args.gradcam_compute_every <= 1 or (step_i % args.gradcam_compute_every == 0))
            )
            if should_compute_cam:
                cam = model.compute_gradcam_from_logits(
                    logits=outputs,
                    target_labels=task_labels,
                    name=args.gradcam_target,
                    retain_graph=True,
                    create_graph=(args.gradcam_loss_lambda > 0),
                )
                if cam is not None:
                    cam_mean = cam.mean()
                    if args.gradcam_loss_lambda > 0:
                        p = cam / (cam.sum(dim=1, keepdim=True) + 1e-6)
                        if args.gradcam_loss_mode == 'entropy':
                            cam_loss = -(p * (p + 1e-6).log()).sum(dim=1).mean()
                        else:  # mass
                            cam_loss = -cam_mean

            loss = (
                ce_loss
                + args.gate_consistency_lambda * gate_loss
                + (args.center_loss_lambda * center_loss if center_criterion is not None else 0.0)
                + args.gradcam_loss_lambda * cam_loss
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        meter.update(loss.item(), task_labels.size(0))
        meter_ce.update(ce_loss.item(), task_labels.size(0))
        meter_gate.update(gate_loss.item(), task_labels.size(0))
        meter_center.update(center_loss.item(), task_labels.size(0))
        meter_cam.update(cam_loss.item(), task_labels.size(0))
        meter_cam_mean.update(cam_mean.item(), task_labels.size(0))
        progbar.add(task_labels.size(0), values=[('loss', loss.item())])

    return {
        'loss': meter.avg,
        'ce_loss': meter_ce.avg,
        'gate_loss': meter_gate.avg,
        'center_loss': meter_center.avg,
        'gradcam_loss': meter_cam.avg,
        'gradcam_mean': meter_cam_mean.avg,
    }


@torch.no_grad()
def evaluate(model, loader, device, epoch, args):
    model.eval()
    correct, total = 0, 0

    for _, _, language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in loader:
        language_tokens = language_tokens.to(device, non_blocking=True)
        mask_idxs = mask_idxs.to(device, non_blocking=True)
        keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
        speaker_labels = speaker_labels.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)

        outputs = model(
            language_tokens,
            mask_idxs,
            keypoint_seqs,
            speaker_labels,
            warmup=False
        )
        _, pred = torch.max(outputs, 1)
        total += task_labels.size(0)
        correct += (pred == task_labels).sum().item()

    return correct / max(total, 1)


def save_checkpoint(path, args, model, optimizer, scaler,
                    epoch, best_acc, best_epoch, use_amp, center_criterion=None):
    ckpt = {
        'model_name': args.model_name,
        'model': model.state_dict(),
        'model_kwargs': model.get_model_kwargs() if hasattr(model, 'get_model_kwargs') else None,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'scaler': scaler.state_dict() if use_amp else None,
        'center_criterion': center_criterion.state_dict() if center_criterion is not None else None,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    os.makedirs(args.checkpoint_save_dir, exist_ok=True)
    if args.epochs_warmup >= args.epochs:
        print(
            f"[WARN] epochs_warmup({args.epochs_warmup}) >= epochs({args.epochs}). "
            "Model may never enter full multimodal training."
        )

    model_kwargs = build_model_kwargs(args)
    model = MultimodalBaseline(**model_kwargs).to(device)
    model.set_gradcam(
        enabled=(args.enable_gradcam or args.gradcam_loss_lambda > 0),
        target_name=args.gradcam_target
    )

    center_criterion = None
    if args.use_center_loss:
        center_criterion = CenterLoss(num_classes=args.max_people_num, feat_dim=512).to(device)

    language_params = [p for n, p in model.named_parameters()
                       if 'convers_encoder' in n]
    other_params = [p for n, p in model.named_parameters()
                    if 'convers_encoder' not in n]
    center_params = list(center_criterion.parameters()) if center_criterion is not None else []

    param_groups = [
        {'params': other_params, 'lr': args.learning_rate * 10},
        {'params': language_params, 'lr': args.learning_rate},
    ]
    if center_params:
        param_groups.append({'params': center_params, 'lr': args.center_loss_lr})
    optimizer = torch.optim.Adam(param_groups)

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer
    collate = partial(collate_fn, tokenizer)
    wandb_run = maybe_init_wandb(args)

    train_dataset = SocialDataset(args, is_training=True)
    test_dataset = SocialDataset(args, is_training=False)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(args.gate_consistency_lambda <= 0.0),
        collate_fn=collate,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=pin
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=pin
    )

    criterion = torch.nn.CrossEntropyLoss()
    if args.use_class_weight:
        class_weight, class_counts = build_class_weight(
            train_dataset, args.max_people_num, args.class_weight_mode
        )
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))
        print(f"[CLASS-WEIGHT] enabled | mode={args.class_weight_mode}")
        print(f"[CLASS-WEIGHT] train_counts={class_counts.astype(int).tolist()}")
        print(f"[CLASS-WEIGHT] weight={class_weight.tolist()}")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        loaded_optimizer = True
        if (
            args.use_film_fusion
            or args.film_vision_layers > 0
            or args.film_fusion_layers > 0
            or ckpt.get('model_kwargs') is not None
            or args.use_center_loss
        ):
            missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
            print(f"[RESUME] strict=False | missing={len(missing)} unexpected={len(unexpected)}")
        else:
            model.load_state_dict(ckpt['model'])
        if center_criterion is not None and ckpt.get('center_criterion'):
            center_criterion.load_state_dict(ckpt['center_criterion'])
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except ValueError as e:
            loaded_optimizer = False
            print(f"[RESUME] optimizer state skipped due to mismatch: {e}")
            print("[RESUME] continue with freshly initialized optimizer.")
        if ckpt.get('scaler') and use_amp:
            scaler.load_state_dict(ckpt['scaler'])
        if loaded_optimizer:
            start_epoch = ckpt.get('epoch', 0) + 1
            best_acc = ckpt.get('best_acc', 0.0)
            best_epoch = ckpt.get('best_epoch', 0)
            print(f"[RESUME] epoch={start_epoch}, best_acc={best_acc:.3f}")
        else:
            # Treat as fine-tuning from pretrained weights with new optimizer/groups.
            start_epoch = 0
            best_acc = 0.0
            best_epoch = 0
            print("[RESUME] start fresh epochs for fine-tuning from loaded model weights.")

    # loop
    if args.gate_consistency_lambda > 0:
        print(
            f"[GATE-TRAIN] enabled | lambda={args.gate_consistency_lambda} "
            f"alpha={args.gate_consistency_alpha} tau={args.gate_consistency_tau}"
        )
        print("[GATE-TRAIN] train_loader shuffle=False for temporal consistency.")

    for epoch in range(start_epoch, args.epochs):
        gate_teacher = None
        if args.gate_consistency_lambda > 0:
            gate_teacher = GateTeacherState(
                num_classes=args.max_people_num,
                alpha=args.gate_consistency_alpha,
                tau=args.gate_consistency_tau
            )

        train_stats = train_one_epoch(
            model, train_loader, optimizer,
            criterion, scaler, device, use_amp, epoch, args,
            gate_teacher=gate_teacher, center_criterion=center_criterion
        )
        train_loss = train_stats['loss']
        train_ce_loss = train_stats['ce_loss']
        train_gate_loss = train_stats['gate_loss']
        test_acc = evaluate(model, test_loader, device, epoch, args)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_checkpoint(
                os.path.join(args.checkpoint_save_dir, 'model.pt'),
                args, model, optimizer, scaler,
                epoch, best_acc, best_epoch, use_amp, center_criterion=center_criterion
            )

        if args.save_last:
            save_checkpoint(
                os.path.join(args.checkpoint_save_dir, 'last.pt'),
                args, model, optimizer, scaler,
                epoch, best_acc, best_epoch, use_amp, center_criterion=center_criterion
            )

        print()
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.3f}")
        if args.gate_consistency_lambda > 0:
            print(f"Train CE: {train_ce_loss:.3f} | Train Gate KL: {train_gate_loss:.3f}")
        if center_criterion is not None:
            print(f"Train Center: {train_stats['center_loss']:.3f} (lambda={args.center_loss_lambda})")
        if args.enable_gradcam or args.gradcam_loss_lambda > 0:
            print(
                f"Grad-CAM mean: {train_stats['gradcam_mean']:.3f} | "
                f"Grad-CAM reg: {train_stats['gradcam_loss']:.3f} "
                f"(lambda={args.gradcam_loss_lambda})"
            )
        print(f"Test Acc: {test_acc:.3f}")
        print(f"Best Acc: {best_acc:.3f} @ {best_epoch+1}")
        print(f"Device: {device} | AMP: {use_amp}")

        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train/loss': train_stats['loss'],
                'train/ce_loss': train_stats['ce_loss'],
                'train/gate_loss': train_stats['gate_loss'],
                'train/center_loss': train_stats['center_loss'],
                'train/gradcam_loss': train_stats['gradcam_loss'],
                'train/gradcam_mean': train_stats['gradcam_mean'],
                'eval/test_acc': test_acc,
                'eval/best_acc': best_acc,
                'eval/best_epoch': best_epoch + 1,
            })

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
