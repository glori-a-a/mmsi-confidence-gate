import torch
import torch.nn as nn
import math
from typing import Dict, Optional

from modules.gaze import GazeEncoder
from modules.gesture import GestureEncoder



class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Works with x shape: (seq_len, batch, d_model)
    """
    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model)) 

        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  

        self.register_buffer("encoding", pe)  

    def forward(self, x: torch.Tensor):
        seq_len = x.size(0)
        return x + self.encoding[:seq_len, :].unsqueeze(1) 


class FiLMBlock(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        # x: (S, B, D), cond_vec: (B, D)
        gb = self.generator(cond_vec)
        gamma, beta = torch.chunk(gb, chunks=2, dim=-1)
        gamma = 1.0 + gamma
        return gamma.unsqueeze(0) * x + beta.unsqueeze(0)


class FiLMStack(nn.Module):
    def __init__(self, num_layers: int, feat_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.blocks = nn.ModuleList(
            [FiLMBlock(feat_dim=feat_dim, hidden_dim=hidden_dim) for _ in range(max(0, int(num_layers)))]
        )

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor):
        for block in self.blocks:
            x = block(x, cond_vec)
        return x


class MultimodalBaseline(nn.Module):
    def __init__(
        self,
        class_num,
        language_model,
        use_film_fusion=False,
        film_vision_layers: int = 0,
        film_fusion_layers: int = 0,
        film_hidden_dim: int = 512,
        enable_gradcam: bool = False,
    ):
        super(MultimodalBaseline, self).__init__()

        self.class_num = class_num
        self.language_model = language_model
        self.use_film_fusion = bool(use_film_fusion)

        # Backward-compatible behavior:
        # old --use_film_fusion meant a single FiLM on the visual branch.
        if self.use_film_fusion and film_vision_layers <= 0 and film_fusion_layers <= 0:
            film_vision_layers = 1

        self.film_vision_layers = max(0, int(film_vision_layers))
        self.film_fusion_layers = max(0, int(film_fusion_layers))
        self.film_hidden_dim = int(film_hidden_dim)

        self.gradcam_enabled = bool(enable_gradcam)
        self.gradcam_target_name = "vision_post_transformer"
        self._gradcam_cache: Dict[str, torch.Tensor] = {}
        self._gradcam_grads: Dict[str, torch.Tensor] = {}

        if language_model == 'bert':
            from transformers import BertModel
            self.convers_encoder = BertModel.from_pretrained('bert-base-uncased')
        elif language_model == 'roberta':
            from transformers import RobertaModel
            self.convers_encoder = RobertaModel.from_pretrained('roberta-base')
        elif language_model == 'electra':
            from transformers import ElectraModel
            self.convers_encoder = ElectraModel.from_pretrained('google/electra-base-discriminator')
        else:
            raise ValueError(f"Unsupported language model: {language_model}")

        self.convers_fc = nn.Sequential(nn.Linear(768, 512))

        self.coordinate_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64)
        )

        self.speaker_encoder = nn.Sequential(
            nn.Linear(9 * 64, 512),
            Permute(*(0, 2, 1)),
            nn.BatchNorm1d(512),
            Permute(*(0, 2, 1)),
            nn.ReLU(),
            nn.Linear(512, 512),
            Permute(*(0, 2, 1)),
            nn.BatchNorm1d(512),
            Permute(*(0, 2, 1)),
            nn.ReLU(),
            nn.Linear(512, 512),
            Permute(*(0, 2, 1)),
            nn.BatchNorm1d(512),
            Permute(*(0, 2, 1)),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

        self.position_encoder = nn.Sequential(
            nn.Linear(6 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.position_fc = nn.Sequential(nn.Linear(512, 512))

        self.onehot_encoder = nn.Sequential(nn.Linear(class_num, 512))

        visual_trans_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.visual_trans = nn.TransformerEncoder(visual_trans_layer, num_layers=3)

        multi_trans_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.multi_trans = nn.TransformerEncoder(multi_trans_layer, num_layers=2)
        self.multi_trans_pre = nn.TransformerEncoder(multi_trans_layer, num_layers=2)

        self.positional_enc = PositionalEncoding(d_model=512, max_len=50)  # 稍微大一点更安全
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))

        
        # speaker_feature_raw shape: (B, 16, 9*64) => in_dim = 9*64
        self.gaze_token_enc = GazeEncoder(in_dim=9 * 64, out_dim=512)
        self.gesture_token_enc = GestureEncoder(in_dim=9 * 64, out_dim=512)

        self.classifier = nn.Sequential(nn.Linear(512, class_num))

        self.vision_film_stack = FiLMStack(
            num_layers=self.film_vision_layers, feat_dim=512, hidden_dim=self.film_hidden_dim
        )
        self.fusion_film_stack = FiLMStack(
            num_layers=self.film_fusion_layers, feat_dim=512, hidden_dim=self.film_hidden_dim
        )

    def get_model_kwargs(self):
        return {
            "class_num": self.class_num,
            "language_model": self.language_model,
            "use_film_fusion": self.use_film_fusion,
            "film_vision_layers": self.film_vision_layers,
            "film_fusion_layers": self.film_fusion_layers,
            "film_hidden_dim": self.film_hidden_dim,
            "enable_gradcam": self.gradcam_enabled,
        }

    def set_gradcam(self, enabled: bool = True, target_name: str = "vision_post_transformer"):
        self.gradcam_enabled = bool(enabled)
        self.gradcam_target_name = str(target_name)

    def clear_gradcam_cache(self):
        self._gradcam_cache.clear()
        self._gradcam_grads.clear()

    def _cache_gradcam_tensor(self, name: str, x: torch.Tensor):
        if not self.gradcam_enabled:
            return x
        self._gradcam_cache[name] = x
        if x.requires_grad:
            x.retain_grad()

            def _save_grad(grad):
                self._gradcam_grads[name] = grad

            x.register_hook(_save_grad)
        return x

    def get_gradcam(self, name: Optional[str] = None):
        key = name or self.gradcam_target_name
        acts = self._gradcam_cache.get(key)
        grads = self._gradcam_grads.get(key)
        if acts is None or grads is None:
            return None
        # Token-wise Grad-CAM for sequence tensors shaped (S, B, D) -> (B, S)
        cam = torch.relu((acts * grads).sum(dim=-1)).transpose(0, 1)
        cam = cam / (cam.max(dim=1, keepdim=True).values + 1e-6)
        return cam

    def compute_gradcam_from_logits(
        self,
        logits: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
        retain_graph: bool = True,
        create_graph: bool = False,
    ):
        key = name or self.gradcam_target_name
        acts = self._gradcam_cache.get(key)
        if acts is None:
            return None
        if target_labels is None:
            target_labels = torch.argmax(logits.detach(), dim=1)
        score = logits.gather(1, target_labels.view(-1, 1)).sum()
        grads = torch.autograd.grad(
            score, acts, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True
        )[0]
        if grads is None:
            return None
        self._gradcam_grads[key] = grads
        return self.get_gradcam(key)

    def forward(
        self,
        language_token,
        mask_idxs,
        keypoint_seqs,
        speaker_labels,
        warmup=False,
        return_aux: bool = False,
    ):
        self.clear_gradcam_cache()
        # encode language conversation
        pad_val = 0 if self.language_model in ['bert', 'electra'] else 1
        attention_mask = (language_token != pad_val).float()
        convers_features = self.convers_encoder(language_token, attention_mask)[0]
        convers_feature = convers_features[torch.arange(len(language_token)), mask_idxs]  # (B,768)
        convers_feature = self.convers_fc(convers_feature).unsqueeze(0)  # (1,B,512)

        # encode speaker non-verbal behaviors
        batch_size = speaker_labels.size(0)
        gaze_feature, gesture_feature = [], []

        for batch_i in range(batch_size):
            spk = speaker_labels[batch_i]
            gaze_feature.append(keypoint_seqs[batch_i:batch_i + 1, spk, :, 0:3 * 2])       # (1,T,6)
            gesture_feature.append(keypoint_seqs[batch_i:batch_i + 1, spk, :, 5 * 2:11 * 2])  # (1,T,12)

        speaker_feature = torch.concat(
            [torch.concat(gaze_feature, dim=0), torch.concat(gesture_feature, dim=0)],
            dim=-1
        )  # (B,T,18)

        # reshape -> coordinate_fc -> (B,16,9*64)
        speaker_feature = speaker_feature.view(batch_size, 16, -1, 2)  # (B,16,9,2)
        speaker_feature_raw = self.coordinate_fc(speaker_feature).view(batch_size, 16, -1)  # (B,16,9*64)

        
        speaker_pool = speaker_feature_raw.mean(dim=1)            # (B,576)

        gaze_vec = self.gaze_token_enc(speaker_pool)             # (B,512)
        gesture_vec = self.gesture_token_enc(speaker_pool)       # (B,512)

        gaze_token = gaze_vec.unsqueeze(0)                       # (1,B,512)
        gesture_token = gesture_vec.unsqueeze(0)                 # (1,B,512)


        
        speaker_feature = self.speaker_encoder(speaker_feature_raw).permute(1, 0, 2)  # (16,B,512)

        # encode listener positions
        speaker_onehot = torch.nn.functional.one_hot(speaker_labels, num_classes=self.class_num).float()
        speaker_onehot_feature = self.onehot_encoder(speaker_onehot)  # (B,512)

        position_feature = keypoint_seqs[:, :, 5, 0:2]  # (B,P,2)
        position_feature = self.coordinate_fc(position_feature).view(batch_size, -1)  # (B, P*64)
        position_feature = self.position_encoder(position_feature) + speaker_onehot_feature  # (B,512)
        position_feature = self.position_fc(position_feature).unsqueeze(0)  # (1,B,512)

        # encode visual interactions
        cls_tokens = self.cls_token.repeat(1, batch_size, 1)  # (1,B,512)

        vis_feature = torch.concat(
            [position_feature, self.positional_enc(speaker_feature[::2, :, :])],
            dim=0
        )  # (1+8,B,512) if speaker_feature is 16 -> [::2] gives 8

        film_cond = convers_feature.squeeze(0)  # (B,512)
        if self.film_vision_layers > 0:
            vis_feature = self.vision_film_stack(vis_feature, film_cond)

        self._cache_gradcam_tensor("vision_pre_transformer", vis_feature)

        vis_feature = self.visual_trans(vis_feature)
        self._cache_gradcam_tensor("vision_post_transformer", vis_feature)
        vis_feature = self.positional_enc(vis_feature)

        # fuse multimodal features
        if warmup:
           
            fused_feature = torch.concat([cls_tokens, gaze_token, gesture_token, vis_feature], dim=0)
            if self.film_fusion_layers > 0:
                fused_feature = self.fusion_film_stack(fused_feature, film_cond)
            fused_feature = self.multi_trans_pre(fused_feature)
        else:
            fused_feature = torch.concat([cls_tokens, convers_feature, gaze_token, gesture_token, vis_feature], dim=0)
            if self.film_fusion_layers > 0:
                fused_feature = self.fusion_film_stack(fused_feature, film_cond)
            fused_feature = self.multi_trans(fused_feature)

        self._cache_gradcam_tensor("fusion_output", fused_feature)

        cls_feature = fused_feature[0, :, :]
        logits = self.classifier(cls_feature)  # (B,class_num)

        if not return_aux:
            return logits
        return {
            "logits": logits,
            "cls_feature": cls_feature,
            "convers_feature": convers_feature.squeeze(0),
            "vis_feature": vis_feature,
            "fused_feature": fused_feature,
        }
