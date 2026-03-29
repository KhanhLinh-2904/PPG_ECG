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
        """
        Args:
            embed_dim: Số chiều của đặc trưng (channel).
            mask_length: Độ dài của mỗi đoạn bị che (span length).
            max_prob: Tỷ lệ che tối đa để tránh trường hợp che mất 100% tín hiệu.
        """
        super().__init__()
        self.mask_length = mask_length
        self.max_prob = max_prob 
        self.mask_emb = nn.Parameter(torch.FloatTensor(embed_dim))
        nn.init.uniform_(self.mask_emb)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Khi validation/test thì không mask
        if not self.training:
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        bsz, seq_len, _ = x.shape

        # ==========================================
        # RANDOM COSINE PROBABILITY GENERATION
        # ==========================================
        # Sinh một góc ngẫu nhiên từ 0 đến Pi/2
        random_angle = torch.rand(1).item() * (math.pi / 2)
        
        # Hàm Cosine sẽ chuyển góc này thành một xác suất từ 0 đến 1
        # Nhân với max_prob để giới hạn mức trần (ví dụ: [0, 0.8])
        current_mask_prob = math.cos(random_angle) * self.max_prob

        # Nếu xác suất quá nhỏ (gần 0), bỏ qua việc mask để tiết kiệm tính toán
        if current_mask_prob < 1e-4:
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        # In ra màn hình để bạn dễ debug theo dõi sự thay đổi (có thể comment lại khi train thật)
        # print(f"Dynamic mask_prob: {current_mask_prob:.4f}, seq_len: {seq_len}")

        # ==========================================
        # VECTORIZED SPAN MASKING
        # ==========================================
        # Tính toán xác suất điểm bắt đầu
        mask_start_prob = current_mask_prob / self.mask_length
        
        # Tung đồng xu cho toàn bộ các điểm trong batch
        mask_starts = torch.rand((bsz, seq_len), device=x.device) < mask_start_prob
        
        # Chặn không cho bắt đầu mask ở đoạn sát cuối
        mask_starts[:, -self.mask_length + 1:] = False
        
        if not mask_starts.any():
            return x, torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)

        # Chuyển đổi sang float và thêm chiều kênh (channel) để đưa qua MaxPool1d
        mask_starts_float = mask_starts.float().unsqueeze(1) 
        
        # Tự đệm (pad) bằng 0 vào bên trái để chiều dài đầu ra khớp chuẩn với seq_len
        padded_starts = F.pad(mask_starts_float, (self.mask_length - 1, 0))
        
        # Kéo dài điểm start thành đoạn dài 'mask_length'
        mask_expanded = F.max_pool1d(
            padded_starts, 
            kernel_size=self.mask_length, 
            stride=1, 
            padding=0
        )
        
        # Ép kiểu về ma trận Boolean
        mask = mask_expanded[:, 0, :].bool()

        # Thay thế bằng Mask Embedding
        x_masked = x.clone() 
        x_masked[mask] = self.mask_emb
        
        return x_masked, mask
    
class StandardVectorQuantizer(nn.Module):
    def __init__(self, dim: int, num_vars: int = 320, groups: int = 2, vq_dim: int = 256, time_first: bool = False, commitment_weight: float = 0.25):
        super().__init__()
        self.groups = groups
        self.num_vars = num_vars
        self.time_first = time_first
        self.commitment_weight = commitment_weight # Hệ số beta cho Commitment Loss
        
        var_dim = vq_dim // groups
        # Khởi tạo Codebook (Từ điển): (1, số nhóm, số từ, kích thước 1 từ)
        self.vars = nn.Parameter(torch.FloatTensor(1, groups, num_vars, var_dim))
        nn.init.uniform_(self.vars, -1 / num_vars, 1 / num_vars)

        self.weight_proj = nn.Linear(dim, groups * var_dim)
        nn.init.normal_(self.weight_proj.weight, mean=0, std=1)
        nn.init.zeros_(self.weight_proj.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.time_first: 
            x = x.transpose(1, 2)
            
        bsz, tsz, fsz = x.shape
        
        # Chiếu và định dạng lại tensor: (Batch * Time, Groups, Var_Dim)
        x = self.weight_proj(x)
        x = x.view(bsz * tsz, self.groups, -1)
        
        q_vec = torch.zeros_like(x)
        encoding_indices = torch.zeros(bsz * tsz, self.groups, dtype=torch.long, device=x.device)
        
        # Duyệt qua từng nhóm để tìm vector gần nhất trong codebook
        for g in range(self.groups):
            # Tính khoảng cách Euclidean L2 giữa x và các vector trong codebook
            # x[:, g, :]: (B*T, var_dim) | self.vars[0, g, :]: (num_vars, var_dim)
            distances = torch.cdist(x[:, g, :], self.vars[0, g, :], p=2.0)
            
            # Chọn index của vector có khoảng cách nhỏ nhất
            idx = torch.argmin(distances, dim=-1)
            encoding_indices[:, g] = idx
            
            # Lấy vector ra từ codebook
            q_vec[:, g, :] = self.vars[0, g, idx]
            
        # ==========================================
        # COMMITMENT LOSS (VQ-VAE)
        # ==========================================
        # q_loss: Ép codebook dịch chuyển về phía đầu ra của CNN
        # e_loss: Ép đầu ra CNN bám sát vào vector trong codebook
        q_loss = F.mse_loss(q_vec.detach(), x)
        e_loss = F.mse_loss(q_vec, x.detach())
        commitment_loss = q_loss + self.commitment_weight * e_loss

        # ==========================================
        # STRAIGHT-THROUGH ESTIMATOR (STE)
        # ==========================================
        # Kỹ thuật copy gradient: Ở lượt đi (Forward) giá trị là q_vec
        # Nhưng ở lượt về (Backward) gradient của q_vec sẽ truyền thẳng sang cho x
        q_vec_st = x + (q_vec - x).detach()
        
        # Trả về shape nguyên bản
        q_vec_st = q_vec_st.view(bsz, tsz, -1)
        targets = encoding_indices.view(bsz, tsz, self.groups).detach()
        
        if not self.time_first: 
            q_vec_st = q_vec_st.transpose(1, 2)
            
        return {
            "q": q_vec_st, 
            "targets": targets, 
            "commitment_loss": commitment_loss
        }

class StandardInfoNCEContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, beta_commitment=1.0):
        """
        beta_commitment: Trọng số để cân bằng giữa Contrastive Loss và Commitment Loss.
        """
        super().__init__()
        self.temperature = temperature
        self.beta_commitment = beta_commitment

    def forward(self, local_reps, q_targets, mask_indices, commitment_loss):
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

        # ==========================================
        # 2. TỔNG LOSS = CONTRASTIVE + COMMITMENT
        # ==========================================
        # Không cần phạt diversity (perplexity) nữa
        total_loss = local_loss + (self.beta_commitment * commitment_loss)
        
        # Print debug để dễ theo dõi quá trình hội tụ
        # print(f"Contrastive Loss: {local_loss.item():.4f} | Commitment Loss: {commitment_loss.item():.4f}")
        
        return total_loss, local_loss, commitment_loss


class ECGTransformerModel(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, conv_layers: List[Tuple[int, int, int]] = [(256, 2, 2)]*4, extractor_mode: str = "layer_norm", conv_bias: bool = False, feature_grad_mult: float = 1.0, vq_dim: int = 256, transformer_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.conv_layers_cfg = conv_layers
        self.cnn_out_dim = conv_layers[-1][0]
        self.feature_grad_mult = feature_grad_mult

        self.feature_extractor = ConvFeatureExtraction(conv_layers=self.conv_layers_cfg, in_d=in_channels, mode=extractor_mode, conv_bias=conv_bias)
        self.layer_norm = nn.LayerNorm(self.cnn_out_dim)
        self.quantizer = StandardVectorQuantizer(dim=self.cnn_out_dim, groups=2, num_vars=320, vq_dim=vq_dim)
        self.post_extract_proj = nn.Linear(self.cnn_out_dim, embed_dim) if self.cnn_out_dim != embed_dim else None
        self.masker = RandomCosineFeatureMasking(embed_dim=embed_dim, mask_length=4, max_prob = 0.8)
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
            "commitment_loss": q_result["commitment_loss"] # SỬA DÒNG NÀY (thay vì perplexity)
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
    batch = 1
    channel = 1
    length = 2400
    ecg_dum = torch.randn(batch, channel, length)
    model = ECGTransformerModel()
    result = model(ecg_dum)
    print("ecg dummy: ", ecg_dum)
