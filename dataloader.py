import json
import re
import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def collate_fn(tokenizer, batch):
    """
    batch elements:
      (file_name, time_sec, tokens, mask_idx, keypoint_seq, speaker_label, task_label)
    """
    file_names, time_secs, tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels = zip(*batch)

    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)

    time_secs = torch.tensor(time_secs, dtype=torch.long)
    mask_idxs = torch.tensor(mask_idxs, dtype=torch.long)
    task_labels = torch.tensor(task_labels, dtype=torch.long)
    speaker_labels = torch.tensor(speaker_labels, dtype=torch.long)

    keypoint_seqs = torch.tensor(np.array(keypoint_seqs), dtype=torch.float32)

    return list(file_names), time_secs, tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels


class SocialDataset(Dataset):
    def __init__(self, args, is_training=True):
        self.is_training = is_training
        self.tokenizer = args.tokenizer
        self.language_model = args.language_model
        self.context_length = args.context_length
        self.txt_labeled_dir = args.txt_labeled_dir
        self.meta_dir = args.meta_dir
        self.keypoint_dir = args.keypoint_dir
        self.task = args.task
        self.txt_dir = args.txt_dir  
