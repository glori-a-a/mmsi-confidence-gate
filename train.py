import argparse
import random
from functools import partial
import os

import numpy as np
import torch
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
    parser.add_argument('--max_people_num', type=int, default=6)
    parser.add_argument('--context_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--epochs_warmup', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)

    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--save_last', action='store_true')

    return parser.parse_args()


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
                    scaler, device, use_amp, epoch, args):
    model.train()
    meter = AverageMeter()
    progbar = Progbar(len(loader.dataset))

    for _, _, language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in loader:
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
            loss = criterion(outputs, task_labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        meter.update(loss.item(), task_labels.size(0))
        progbar.add(args.batch_size, values=[('loss', loss.item())])

    return meter.avg


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
            warmup=(epoch < args.epochs_warmup)
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

    model = MultimodalBaseline(
        args.max_people_num,
        args.language_model
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
        shuffle=True,
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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ckpt.get('scaler') and use_amp:
            scaler.load_state_dict(ckpt['scaler'])

        start_epoch = ckpt.get('epoch', 0) + 1
        best_acc = ckpt.get('best_acc', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)

        print(f"[RESUME] epoch={start_epoch}, best_acc={best_acc:.3f}")

    # loop
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, scaler, device, use_amp, epoch, args
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
        print(f"Test Acc: {test_acc:.3f}")
        print(f"Best Acc: {best_acc:.3f} @ {best_epoch+1}")
        print(f"Device: {device} | AMP: {use_amp}")


if __name__ == '__main__':
    main()
