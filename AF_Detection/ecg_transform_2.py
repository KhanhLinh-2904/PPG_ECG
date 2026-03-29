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

class RandomCosineFeatureMasking(nn.Module):
    def __init__(self, embed_dim: int, mask_length: int = 10, max_prob: float = 0.8):
        super().__init__()
        self.mask_length = mask_length
        self.max_prob = max_prob 
        self.mask_emb = nn.Parameter(torch.FloatTensor(embed_dim))
        nn.init.uniform_(self.mask_emb)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        bsz, seq_len, _ = x.shape
        random_angle = torch.rand(1).item() * (math.pi / 2)
        current_mask_prob = math.cos(random_angle) * self.max_prob

        if current_mask_prob < 1e-4:
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        mask_start_prob = current_mask_prob / self.mask_length
        mask_starts = torch.rand((bsz, seq_len), device=x.device) < mask_start_prob
        mask_starts[:, -self.mask_length + 1:] = False
        
        if not mask_starts.any():
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        mask_starts_float = mask_starts.float().unsqueeze(1) 
        padded_starts = F.pad(mask_starts_float, (self.mask_length - 1, 0))
        
        mask_expanded = F.max_pool1d(
            padded_starts, 
            kernel_size=self.mask_length, 
            stride=1, 
            padding=0
        )
        mask = mask_expanded[:, 0, :].bool()

        x_masked = x.clone() 
        x_masked[mask] = self.mask_emb
        
        return x_masked, mask

# ==========================================
# HUBERT LOSS (Thay thế cho Contrastive Loss)
# ==========================================
class HuBERTCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, target_labels: torch.Tensor, mask_indices: torch.Tensor):
        """
        logits: Đầu ra của mô hình (Batch, Time, Num_Clusters)
        target_labels: Nhãn K-means được tạo offline (Batch, Time)
        mask_indices: Ma trận boolean chỉ định vị trí bị mask (Batch, Time)
        """
        # Chỉ lấy các vị trí đã bị che để tính Loss
        logits_masked = logits[mask_indices]       
        targets_masked = target_labels[mask_indices] 
        
        if logits_masked.size(0) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return F.cross_entropy(logits_masked, targets_masked)

# ==========================================
# MÔ HÌNH BACKBONE THEO PHONG CÁCH HUBERT
# ==========================================
class ECGTransformerModel(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, num_clusters: int = 100, conv_layers: List[Tuple[int, int, int]] = [(256, 2, 2)]*4, extractor_mode: str = "layer_norm", conv_bias: bool = False, feature_grad_mult: float = 1.0, transformer_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.conv_layers_cfg = conv_layers
        self.cnn_out_dim = conv_layers[-1][0]
        self.feature_grad_mult = feature_grad_mult
        self.num_clusters = num_clusters

        self.feature_extractor = ConvFeatureExtraction(conv_layers=self.conv_layers_cfg, in_d=in_channels, mode=extractor_mode, conv_bias=conv_bias)
        self.layer_norm = nn.LayerNorm(self.cnn_out_dim)
        
        # Đã loại bỏ Quantizer tại đây
        
        self.post_extract_proj = nn.Linear(self.cnn_out_dim, embed_dim) if self.cnn_out_dim != embed_dim else None
        self.masker = RandomCosineFeatureMasking(embed_dim=embed_dim, mask_length=4, max_prob=0.8)
        self.conv_pos = ConvPositionalEncoding(embed_dim=embed_dim, kernel_size=128, groups=16)
        self.encoder = transformer_encoder
        
        # Thêm Lớp chiếu ra số lượng cụm K-means để làm bài toán phân loại nhãn giả
        self.label_proj = nn.Linear(embed_dim, self.num_clusters)

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

        # Chiếu ra kích thước K-means clusters
        logits = self.label_proj(local_reps)

        return {
            "local_reps": local_reps,     # Vẫn trả về để dùng cho fine-tuning Classifier sau này
            "logits": logits,             # Trả về để dùng tính loss ở giai đoạn Pre-train
            "mask_indices": mask_indices,
            "padding_mask": new_padding_mask
        }

class ECGAFClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int = 256, hidden_dim: int = 128, num_classes: int = 2, dropout_prob: float = 0.1, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(hidden_dim // 4, num_classes)
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
    
if __name__ == "__main__":
    # Test thử kích thước đầu ra
    batch = 2
    channel = 1
    length = 2400
    ecg_dum = torch.randn(batch, channel, length)
    
    # Pre-train thử
    model = ECGTransformerModel(num_clusters=100)
    result = model(ecg_dum)
    
    print("=== HUBERT PRE-TRAINING OUTPUT ===")
    print("Logits K-Means shape:", result["logits"].shape) # Mong đợi: (Batch, Time_Steps, Num_Clusters)
    print("Mask Indices shape:", result["mask_indices"].shape)
    
    # Giả lập nhãn được tạo từ K-means offline
    dummy_kmeans_labels = torch.randint(0, 100, (batch, result["logits"].shape[1]))
    
    # Test hàm Loss
    criterion = HuBERTCrossEntropyLoss()
    loss = criterion(result["logits"], dummy_kmeans_labels, result["mask_indices"])
    print(f"\nHuBERT Pre-train Loss: {loss.item():.4f}")
    
    # Test thử với Classifier
    classifier = ECGAFClassifier(backbone=model)
    cls_out = classifier(ecg_dum)
    print("\n=== CLASSIFIER FINE-TUNING OUTPUT ===")
    print("Classifier Logits shape:", cls_out.shape) # Mong đợi: (Batch, 2)