import os
import json
import re
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
    _LINE_RE = re.compile(r"^\s*([^()]+?)\s*\((\d{1,2}):(\d{2})\):\s*(.*)$")
    _TO_PLAYER_RE = re.compile(r"\[To\s+Player(\d+)\]", re.IGNORECASE)
    _PLAYER_RE = re.compile(r"Player(\d+)", re.IGNORECASE)

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
        self.data_split_file = args.data_split_file
        self.max_people_num = getattr(args, 'max_people_num', 6)

        with open(self.data_split_file, "r") as f:
            split = json.load(f)

        # pick split
        if is_training:
            pref = ["train", "training"]
        else:
            pref = ["test", "val", "validation"]

        split_key = None
        for k in pref:
            if k in split:
                split_key = k
                break
        if split_key is None:
            raise KeyError(f"Cannot find split key in {self.data_split_file}. Keys={list(split.keys())}")

        self.ids = split[split_key]
        if not isinstance(self.ids, list):
            raise TypeError(f"Split '{split_key}' must be a list, got {type(self.ids)}")

        # load meta index if exists
        meta_index_path = os.path.join(self.meta_dir, "meta_data.json")
        self.meta_index = None
        if os.path.isfile(meta_index_path):
            with open(meta_index_path, "r") as f:
                self.meta_index = json.load(f)

        self.kp_cache = {}
        self.samples = []
        for sid in self.ids:
            self.samples.extend(self._build_samples_from_sid(sid))

    def _load_meta_for_id(self, sid):
        if self.meta_index is not None and sid in self.meta_index:
            m = self.meta_index[sid]
            if isinstance(m, dict):
                return m
            if isinstance(m, list) and len(m) > 0 and isinstance(m[0], dict):
                return m[0]

        cand = os.path.join(self.meta_dir, f"{sid}.json")
        if os.path.isfile(cand):
            with open(cand, "r") as f:
                return json.load(f)
        return {}

    def __len__(self):
        return len(self.samples)

    def _get_first(self, d, keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    def _split_sid_game(self, sid):
        if "_Game" not in sid:
            return sid, None
        base, game_num = sid.rsplit("_Game", 1)
        try:
            return base, f"Game{int(game_num)}"
        except Exception:
            return base, None

    def _load_player_name_map(self, sid):
        video_id, game_key = self._split_sid_game(sid)
        meta_path = os.path.join(self.meta_dir, f"{video_id}.json")
        if not os.path.isfile(meta_path) or game_key is None:
            return {}

        try:
            meta = json.load(open(meta_path, "r"))
            game = meta.get(game_key, {})
            names = game.get("playerNames", [])
        except Exception:
            return {}

        mapping = {}
        for i, name in enumerate(names):
            mapping[str(name).strip().lower()] = i
        return mapping

    def _speaker_to_idx(self, speaker_name, name_map):
        name = str(speaker_name).strip().lower()
        if name in name_map:
            return int(name_map[name])
        m = self._PLAYER_RE.search(speaker_name)
        if m:
            return int(m.group(1))
        return 0

    def _strip_target_tag(self, utter):
        # Prevent label leakage: remove annotation tags like "[To Player3]"
        cleaned = self._TO_PLAYER_RE.sub("", utter)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _build_samples_from_sid(self, sid):
        txt_path = os.path.join(self.txt_labeled_dir, f"{sid}.txt")
        if not os.path.isfile(txt_path):
            return []

        name_map = self._load_player_name_map(sid)
        samples = []
        history = []

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = self._LINE_RE.match(line.strip())
                if not m:
                    continue

                speaker, mm, ss, utter = m.group(1), m.group(2), m.group(3), m.group(4)
                time_sec = int(mm) * 60 + int(ss)

                # Build context from the recent utterances, but remove target tags
                # so the model cannot read the answer from annotation text.
                utter_clean = self._strip_target_tag(utter)
                history.append(f"{speaker}: {utter_clean}")
                context = history[-max(int(self.context_length), 1):]
                text = " ".join(context)

                enc = self.tokenizer(text, truncation=True, max_length=256, add_special_tokens=True)
                tokens = enc["input_ids"]
                if not tokens:
                    tokens = [self.tokenizer.pad_token_id]

                to_player = self._TO_PLAYER_RE.search(utter)
                if self.task == "STI" and to_player is None:
                    continue
                task_label = int(to_player.group(1)) if to_player else 0

                samples.append({
                    "sid": sid,
                    "file_name": sid,
                    "time_sec": time_sec,
                    "tokens": tokens,
                    "mask_idx": 0,
                    "speaker_label": self._speaker_to_idx(speaker, name_map),
                    "task_label": task_label,
                })

        return samples

    def _get_keypoint_frames(self, sid):
        if sid in self.kp_cache:
            return self.kp_cache[sid]

        kp_path = os.path.join(self.keypoint_dir, f"{sid}.npy")
        if not os.path.isfile(kp_path):
            self.kp_cache[sid] = []
            return self.kp_cache[sid]

        try:
            arr = np.load(kp_path, allow_pickle=True)
            frames = arr.tolist() if isinstance(arr, np.ndarray) else []
            if not isinstance(frames, list):
                frames = []
        except Exception:
            frames = []

        self.kp_cache[sid] = frames
        return frames

    def _load_keypoints(self, sid, time_sec):
        # model expects (P,16,34)
        P = int(getattr(self, 'max_people_num', 6))
        T_max = 16
        D_max = 34
        out = np.zeros((P, T_max, D_max), dtype=np.float32)

        frames = self._get_keypoint_frames(sid)
        if not frames:
            return out

        # Approximate timestamp alignment with 30 fps.
        center = max(0, int(time_sec) * 30)
        start = max(0, center - (T_max - 1))

        for t in range(T_max):
            fi = start + t
            if fi >= len(frames):
                break
            dets = frames[fi]
            if not isinstance(dets, list):
                continue

            for det in dets:
                if not isinstance(det, dict):
                    continue
                idx = det.get("idx", None)
                if idx is None:
                    continue
                try:
                    p = int(idx)
                except Exception:
                    continue
                if p < 0 or p >= P:
                    continue

                kps = det.get("keypoints", [])
                if not isinstance(kps, list) or len(kps) < 2:
                    continue

                # keypoints are [x, y, score] * 17 -> keep x,y only => 34 dims
                xy = np.asarray(kps, dtype=np.float32).reshape(-1, 3)[:, :2].reshape(-1)
                d = min(D_max, xy.size)
                out[p, t, :d] = xy[:d]

        return out

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sid = sample["sid"]
        keypoint_seq = self._load_keypoints(sid, sample["time_sec"])
        return (
            sample["file_name"],
            int(sample["time_sec"]),
            sample["tokens"],
            int(sample["mask_idx"]),
            keypoint_seq,
            int(sample["speaker_label"]),
            int(sample["task_label"]),
        )
