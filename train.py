import argparse
import random
from functools import partial
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
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
    parser.add_argument('--use_class_weight', action='store_true',
                        help='Use class-weighted cross entropy based on train label distribution.')
    parser.add_argument('--class_weight_mode', type=str, default='sqrt_inv',
                        choices=['inv', 'sqrt_inv'],
                        help='Weight mode: inv=1/freq, sqrt_inv=1/sqrt(freq).')

    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--save_last', action='store_true')

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



def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, use_amp, epoch, args, gate_teacher=None):
    model.train()
    meter = AverageMeter()
    meter_ce = AverageMeter()
    meter_gate = AverageMeter()
    progbar = Progbar(len(loader.dataset))

    for file_names, _, language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in loader:
        language_tokens = language_tokens.to(device, non_blocking=True)
        mask_idxs = mask_idxs.to(device, non_blocking=True)
        keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
        speaker_labels = speaker_labels.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            outputs = model(
                language_tokens,
                mask_idxs,
                keypoint_seqs,
                speaker_labels,
                warmup=(epoch < args.epochs_warmup)
            )
            ce_loss = criterion(outputs, task_labels)
            gate_loss = torch.tensor(0.0, device=device)

            if gate_teacher is not None and args.gate_consistency_lambda > 0:
                probs = torch.softmax(outputs, dim=1)
                targets, uncertain_mask = gate_teacher.get_targets_and_update(
                    file_names, probs.detach()
                )
                if uncertain_mask.any():
                    log_probs_u = F.log_softmax(outputs[uncertain_mask], dim=1)
                    targets_u = targets[uncertain_mask]
                    gate_loss = F.kl_div(log_probs_u, targets_u, reduction='batchmean')

            loss = ce_loss + args.gate_consistency_lambda * gate_loss

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
        progbar.add(task_labels.size(0), values=[('loss', loss.item())])

    return meter.avg, meter_ce.avg, meter_gate.avg


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
                    epoch, best_acc, best_epoch, use_amp):
    ckpt = {
        'model_name': args.model_name,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'scaler': scaler.state_dict() if use_amp else None,
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

    model = MultimodalBaseline(
        args.max_people_num,
        args.language_model,
        use_film_fusion=args.use_film_fusion
    ).to(device)

    language_params = [p for n, p in model.named_parameters()
                       if 'convers_encoder' in n]
    other_params = [p for n, p in model.named_parameters()
                    if 'convers_encoder' not in n]

    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.learning_rate * 10},
        {'params': language_params, 'lr': args.learning_rate},
    ])

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer
    collate = partial(collate_fn, tokenizer)

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
        if args.use_film_fusion:
            missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
            print(f"[RESUME-FiLM] strict=False | missing={len(missing)} unexpected={len(unexpected)}")
        else:
            model.load_state_dict(ckpt['model'])
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

        train_loss, train_ce_loss, train_gate_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, scaler, device, use_amp, epoch, args, gate_teacher=gate_teacher
        )
        test_acc = evaluate(model, test_loader, device, epoch, args)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_checkpoint(
                os.path.join(args.checkpoint_save_dir, 'model.pt'),
                args, model, optimizer, scaler,
                epoch, best_acc, best_epoch, use_amp
            )

        if args.save_last:
            save_checkpoint(
                os.path.join(args.checkpoint_save_dir, 'last.pt'),
                args, model, optimizer, scaler,
                epoch, best_acc, best_epoch, use_amp
            )

        print()
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.3f}")
        if args.gate_consistency_lambda > 0:
            print(f"Train CE: {train_ce_loss:.3f} | Train Gate KL: {train_gate_loss:.3f}")
        print(f"Test Acc: {test_acc:.3f}")
        print(f"Best Acc: {best_acc:.3f} @ {best_epoch+1}")
        print(f"Device: {device} | AMP: {use_amp}")


if __name__ == '__main__':
    main()
