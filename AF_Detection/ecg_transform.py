import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

class TransposeLast(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(-2, -1)

class SamePad(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x

class GradMultiply(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.clone()
    @staticmethod 
    def backward(ctx, grad: torch.Tensor):
        return grad * ctx.scale, None

class ConvFeatureExtraction(nn.Module):
    def __init__(self, conv_layers: List[Tuple[int, int, int]], in_d: int = 1, dropout: float = 0.0, mode: str = "default", conv_bias: bool = False):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        current_in_c = in_d
        
        for i, (dim, kernel, stride) in enumerate(conv_layers):
            use_layer_norm = (mode == "layer_norm")
            use_group_norm = (i == 0)
            
            conv = nn.Conv1d(current_in_c, dim, kernel, stride=stride, bias=conv_bias)
            nn.init.kaiming_normal_(conv.weight)
            
            layers = [conv, nn.Dropout(p=dropout)]
            if use_layer_norm:
                layers.extend([TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()])
            elif use_group_norm:
                layers.append(nn.GroupNorm(num_groups=dim, num_channels=dim, affine=True))
                
            layers.append(nn.GELU())
            self.conv_layers.append(nn.Sequential(*layers))
            current_in_c = dim 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2: x = x.unsqueeze(1)
        for conv_block in self.conv_layers: x = conv_block(x)
        return x

class ConvPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, groups: int):
        super().__init__()
        conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=groups)
        nn.init.normal_(conv.weight, mean=0, std=math.sqrt(4.0 / (kernel_size * embed_dim)))
        nn.init.constant_(conv.bias, 0)
        
        try:
            conv = nn.utils.parametrizations.weight_norm(conv, name="weight", dim=2)
        except AttributeError:
            conv = nn.utils.weight_norm(conv, name="weight", dim=2)
            
        self.pos_conv = nn.Sequential(conv, SamePad(kernel_size), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pos_conv(x.transpose(1, 2)).transpose(1, 2)

class GumbelVectorQuantizer(nn.Module):
    def __init__(self, dim, num_vars=320, temp=(2.0, 0.5, 0.999995), groups=2, combine_groups=False, vq_dim=256, time_first=False):
        super().__init__()
        self.groups = groups
        self.num_vars = num_vars
        self.time_first = time_first
        self.num_updates = 0
        
        var_dim = vq_dim // groups
        self.vars = nn.Parameter(torch.FloatTensor(1, groups * num_vars, var_dim))
        nn.init.uniform_(self.vars)

        self.weight_proj = nn.Linear(dim, groups * num_vars)
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

        self.max_temp, self.min_temp, self.temp_decay = temp
        self.curr_temp = self.max_temp

    def set_num_updates(self, num_updates):
        self.curr_temp = max(self.max_temp * self.temp_decay ** num_updates, self.min_temp)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.time_first: x = x.transpose(1, 2)
        bsz, tsz, fsz = x.shape

        x = self.weight_proj(x).view(bsz * tsz * self.groups, -1)

        
        _, k = x.max(-1)
        hard_x = x.new_zeros(*x.shape).scatter_(-1, k.view(-1, 1), 1.0).view(bsz * tsz, self.groups, -1)

        hard_probs = torch.mean(hard_x.float(), dim=0)
        code_perplexity = torch.exp(-torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)).sum()

        if self.training:
            x_idx = F.gumbel_softmax(x.float(), tau=self.curr_temp, hard=True).type_as(x)
        else:
            x_idx = hard_x.view(bsz * tsz * self.groups, -1)

        x_idx = x_idx.view(bsz * tsz, -1).unsqueeze(-1)
        q_vec = (x_idx * self.vars).view(bsz * tsz, self.groups, self.num_vars, -1).sum(-2)
        q_vec = q_vec.view(bsz, tsz, -1)
        
        targets = x.view(bsz * tsz * self.groups, -1).argmax(dim=-1).view(bsz, tsz, self.groups).detach()

        if not self.time_first: q_vec = q_vec.transpose(1, 2)
        return {"q": q_vec, "targets": targets, "perplexity": code_perplexity}

class FeatureMasking(nn.Module):
    def __init__(self, embed_dim: int, mask_prob: float = 0.065, mask_length: int = 10):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_emb = nn.Parameter(torch.FloatTensor(embed_dim))
        nn.init.uniform_(self.mask_emb)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.mask_prob == 0.0:
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        bsz, seq_len, _ = x.shape
        num_mask_spans = int(self.mask_prob * seq_len / self.mask_length)
        num_mask_spans = max(1, num_mask_spans) if self.mask_prob > 0.0 else 0
        # print("num mask span: ", num_mask_spans)
        if num_mask_spans == 0:
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=x.device)
        for b in range(bsz):
            start_indices = torch.randperm(seq_len - self.mask_length, device=x.device)[:num_mask_spans]
            for idx in start_indices:
                mask[b, idx : idx + self.mask_length] = True

        x = x.clone() 
        x[mask] = self.mask_emb
        return x, mask

class ECGTransformerModel(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, conv_layers: List[Tuple[int, int, int]] = [(256, 2, 2)]*4, extractor_mode: str = "layer_norm", conv_bias: bool = False, feature_grad_mult: float = 1.0, vq_dim: int = 256, transformer_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.conv_layers_cfg = conv_layers
        self.cnn_out_dim = conv_layers[-1][0]
        self.feature_grad_mult = feature_grad_mult

        self.feature_extractor = ConvFeatureExtraction(conv_layers=self.conv_layers_cfg, in_d=in_channels, mode=extractor_mode, conv_bias=conv_bias)
        self.layer_norm = nn.LayerNorm(self.cnn_out_dim)
        self.quantizer = GumbelVectorQuantizer(dim=self.cnn_out_dim, groups=2, num_vars=320, vq_dim=vq_dim)
        self.post_extract_proj = nn.Linear(self.cnn_out_dim, embed_dim) if self.cnn_out_dim != embed_dim else None
        self.masker = FeatureMasking(embed_dim=embed_dim, mask_prob=0.4, mask_length=4)
        self.conv_pos = ConvPositionalEncoding(embed_dim=embed_dim, kernel_size=128, groups=16)
        self.encoder = transformer_encoder

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        lengths = input_lengths.float()
        for _, kernel_size, stride in self.conv_layers_cfg:
            lengths = torch.floor((lengths - kernel_size) / stride + 1)
        return lengths.long()

    def forward(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if self.feature_grad_mult > 0:
            latent_features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                latent_features = GradMultiply.apply(latent_features, self.feature_grad_mult)
        else:
            with torch.no_grad(): latent_features = self.feature_extractor(source)

        q_result = self.quantizer(latent_features)
        q_targets = q_result["q"]

        features = self.layer_norm(latent_features.transpose(1, 2))

        new_padding_mask = None
        if padding_mask is not None and padding_mask.any():
            input_lengths = (~padding_mask).sum(dim=-1)
            if input_lengths.dim() > 1: input_lengths = input_lengths[:, 0]
            out_lengths = self._compute_output_lengths(input_lengths)
            max_len = features.size(1)
            indices = torch.arange(max_len, device=features.device).expand(features.size(0), max_len)
            new_padding_mask = indices >= out_lengths.unsqueeze(1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        masked_features, mask_indices = self.masker(features)
        masked_features = masked_features + self.conv_pos(masked_features)

        if self.encoder is not None:
            encoder_out = self.encoder(masked_features, src_key_padding_mask=new_padding_mask)
            if isinstance(encoder_out, tuple): local_reps = encoder_out[0]
            elif isinstance(encoder_out, dict): local_reps = encoder_out["x"]
            else: local_reps = encoder_out
        else:
            local_reps = masked_features 

        return {
            "local_reps": local_reps,
            "q_targets": q_targets.transpose(1, 2),
            "mask_indices": mask_indices,
            "padding_mask": new_padding_mask,         
            "perplexity": q_result["perplexity"]
        }

class ECGAFClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int = 256, hidden_dim: int = 128, num_classes: int = 2, dropout_prob: float = 0.1, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, source: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        backbone_out = self.backbone(source, padding_mask=padding_mask)
        local_reps = backbone_out["local_reps"]
        cnn_padding_mask = backbone_out["padding_mask"]

        if cnn_padding_mask is not None and cnn_padding_mask.any():
            local_reps[cnn_padding_mask] = 0.0
            valid_lengths = (~cnn_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            global_reps = local_reps.sum(dim=1) / valid_lengths
        else:
            global_reps = local_reps.mean(dim=1) 

        logits = self.classifier(global_reps)
        return logits
    
class InfoNCEContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, diversity_weight=0.1, num_vars=320):
        super().__init__()
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.num_vars = num_vars # Số lượng mã từ điển (để tính trần perplexity)

    def forward(self, local_reps, q_targets, mask_indices, perplexity):
        # ==========================================
        # 1. LOCAL CONTRASTIVE LOSS (Bắt chước Q)
        # ==========================================
        c = local_reps[mask_indices]  
        q = q_targets[mask_indices]   

        if c.size(0) == 0:
            local_loss = torch.tensor(0.0, device=c.device, requires_grad=True)
        else:
            c = F.normalize(c, p=2, dim=-1)
            q = F.normalize(q, p=2, dim=-1)
            logits_local = torch.matmul(c, q.transpose(0, 1)) / self.temperature
            labels_local = torch.arange(c.size(0), device=c.device)
            local_loss = F.cross_entropy(logits_local, labels_local)


        diversity_penalty = (self.num_vars - perplexity) / self.num_vars
        print("local_loss: ", local_loss)
        print("diversity_penalty: ", diversity_penalty)

       
        total_loss = local_loss  + (self.diversity_weight * diversity_penalty)
        
        return total_loss, local_loss