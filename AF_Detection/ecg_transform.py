import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
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

class VariationalVectorQuantizer(nn.Module):
    def __init__(self, dim: int, num_vars: int = 320, commitment_weight: float = 0.25, kl_weight: float = 0.1):
        super().__init__()
        self.num_vars = num_vars
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.kl_weight = kl_weight

        self.codebook = nn.Parameter(torch.FloatTensor(num_vars, dim))
        nn.init.uniform_(self.codebook, -1.0 / num_vars, 1.0 / num_vars)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        dist = torch.cdist(x_flat, self.codebook, p=2.0) ** 2
        probs = F.softmax(-dist, dim=-1)

        z_soft = torch.matmul(probs, self.codebook).view(B, T, D)
        indices = torch.argmax(probs, dim=-1) # shape: (B*T,)
        z_hard = F.embedding(indices, self.codebook).view(B, T, D)

     
        encodings = F.one_hot(indices, num_classes=self.num_vars).float()
        
        avg_probs_hard = torch.mean(encodings, dim=0)
        
        entropy = -torch.sum(avg_probs_hard * torch.log(avg_probs_hard + 1e-10))
        perplexity = torch.exp(entropy)
        # =====================================================================

        # Straight-Through Estimator (STE) 
        z_q = z_soft + (z_hard - z_soft).detach()

        if self.training:
            # Codebook Loss & Commitment Loss
            codebook_loss = F.mse_loss(z_q, x.detach())
            commitment_loss = F.mse_loss(x, z_q.detach())

            # KL Divergence 
            avg_probs = torch.mean(probs, dim=0) # Dùng soft probs cho KL Div như cũ
            kl_div = torch.sum(avg_probs * torch.log(avg_probs * self.num_vars + 1e-10))

            vq_loss = codebook_loss + (self.commitment_weight * commitment_loss) + (self.kl_weight * kl_div)
        else:
            vq_loss = torch.tensor(0.0, device=x.device)

        return {
            "q": z_q, 
            "vq_loss": vq_loss, 
            "indices": indices.view(B, T),
            "perplexity": perplexity 
        }

class ECGTransformerModel(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, conv_layers: List[Tuple[int, int, int]] = [(256, 2, 2)]*4, extractor_mode: str = "layer_norm", conv_bias: bool = False, feature_grad_mult: float = 1.0, vq_dim: int = 256, transformer_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.conv_layers_cfg = conv_layers
        self.cnn_out_dim = conv_layers[-1][0]
        self.feature_grad_mult = feature_grad_mult

        self.feature_extractor = ConvFeatureExtraction(conv_layers=self.conv_layers_cfg, in_d=in_channels, mode=extractor_mode, conv_bias=conv_bias)
        self.layer_norm = nn.LayerNorm(self.cnn_out_dim)
        
        self.quantizer = VariationalVectorQuantizer(dim=self.cnn_out_dim, num_vars=320)
        
        self.post_extract_proj = nn.Linear(self.cnn_out_dim, embed_dim) if self.cnn_out_dim != embed_dim else None
        self.masker = RandomCosineFeatureMasking(embed_dim=embed_dim, mask_length=10, max_prob=0.5)
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

        latent_features_t = latent_features.transpose(1, 2)
        
        q_result = self.quantizer(latent_features_t)
        q_targets = q_result["q"]
        vq_loss = q_result["vq_loss"]
        features = self.layer_norm(latent_features_t)


        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        masked_features, mask_indices = self.masker(features)
        masked_features = masked_features + self.conv_pos(masked_features)
        if self.encoder is not None:
            encoder_out = self.encoder(masked_features)
            local_reps = encoder_out
        else:
            local_reps = masked_features 

        return {
            "local_reps": local_reps,
            "q_targets": q_targets, 
            "mask_indices": mask_indices,     
            "vq_loss": vq_loss,
            "perplexity": q_result["perplexity"]
        }


# class ECGAFClassifier(nn.Module):
#     def __init__(self, backbone: nn.Module, embed_dim: int = 256, hidden_dim: int = 128, num_classes: int = 2, dropout_prob: float = 0.1, freeze_backbone: bool = False):
#         super().__init__()
#         self.backbone = backbone
        
#         if freeze_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         self.classifier = nn.Sequential(
          

#             nn.Linear(embed_dim, num_classes),
#             nn.BatchNorm1d(num_classes),
#             nn.ReLU(),
#             nn.Dropout(dropout_prob),

#             # nn.Linear(hidden_dim // 2, num_classes)

#         )
#     def forward(self, source: torch.Tensor) -> torch.Tensor:
#         backbone_out = self.backbone(source)
#         local_reps = backbone_out["local_reps"]
#         global_reps = local_reps.mean(dim=1) 
#         logits = self.classifier(global_reps)
#         return logits

class VQContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, vq_weight=1.0, plot_matrix=True):
        super().__init__()
        self.temperature = temperature
        self.vq_weight = vq_weight 
        self.plot_matrix = plot_matrix

    def forward(self, local_reps, q_targets, mask_indices, vq_loss, vq_perplexity=None):
        c = local_reps[mask_indices]  
        q = q_targets[mask_indices].detach()   

        if c.size(0) == 0:
            local_loss = torch.tensor(0.0, device=c.device, requires_grad=True)
            total_loss = local_loss + (self.vq_weight * vq_loss)
            return total_loss, local_loss

        c = F.normalize(c, p=2, dim=-1)
        q = F.normalize(q, p=2, dim=-1)
        
        logits = torch.matmul(c, q.transpose(0, 1)) / self.temperature
        
        # =====================================================================
        # =====================================================================
        # if self.plot_matrix:
        #     probs = F.softmax(logits, dim=-1)
        #     entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        #     contrastive_perplexity = torch.exp(entropy).item()
            
        #     logits_np = logits.detach().cpu().numpy()
            
        #     plt.figure(figsize=(10, 8))
        #     plt.imshow(logits_np, cmap='viridis')
        #     plt.colorbar(label='Similarity Score (Scaled by Temperature)')
            
        #     num_samples = c.size(0)
        #     title_str = f'Contrastive Logits Matrix (Masked size: {num_samples})\n'
        #     title_str += f'Contrastive Perplexity: {contrastive_perplexity:.2f} / {num_samples}'
            
        #     if vq_perplexity is not None:
        #         vq_perp_val = vq_perplexity.item() if isinstance(vq_perplexity, torch.Tensor) else vq_perplexity
        #         title_str += f' | VQ Codebook Perplexity: {vq_perp_val:.2f}'
                
        #     title_str += '\n( Positive Pairs should have higher similarity )'
            
        #     plt.title(title_str)
        #     plt.xlabel('Codebook Segments Target (q)')
        #     plt.ylabel('Masked Hidden Segments (c)')
            
          
        #     plt.draw()        
        #     plt.pause(3.0)   
        #     plt.close()      
        # # =====================================================================

        labels = torch.arange(c.size(0), device=c.device)
        
        local_loss = F.cross_entropy(logits, labels)

        total_loss = local_loss + (self.vq_weight * vq_loss)
        
        return total_loss, local_loss
    

# class VQSupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.1, vq_weight=1.0):
#         super().__init__()
#         self.temperature = temperature
#         self.vq_weight = vq_weight

#     def forward(self, local_reps, q_targets, mask_indices, vq_loss, batch_labels):
#         c = local_reps[mask_indices]  
#         q = q_targets[mask_indices].detach() 

#         if c.size(0) == 0:
#             return vq_loss * self.vq_weight, torch.tensor(0.0)

#         c = F.normalize(c, p=2, dim=-1)
#         q = F.normalize(q, p=2, dim=-1)

#         logits = torch.matmul(c, q.transpose(0, 1)) / self.temperature

       
#         B, T = local_reps.shape[0], local_reps.shape[1]
#         expanded_labels = batch_labels.unsqueeze(1).expand(B, T)
#         mask_labels = expanded_labels[mask_indices] # Shape [N]

#         # Tạo ma trận nhãn mục tiêu (N x N)
#         # mask_labels[:, None] == mask_labels[None, :] tạo ra ma trận True/False
#         # Chỗ nào cùng nhãn (AF-AF hoặc nonAF-nonAF) sẽ là 1, khác nhãn là 0
#         ground_truth = (mask_labels[:, None] == mask_labels[None, :]).float()

#         # 5. TÍNH LOSS
#         # Thay vì CrossEntropy (chỉ chọn 1 đáp án đúng), ta dùng log_softmax 
#         # và tính trung bình trên tất cả các đáp án cùng nhãn
#         log_probs = F.log_softmax(logits, dim=1)
        
#         # Nhân ma trận log_probs với ground_truth để chỉ giữ lại các cặp cùng nhãn
#         # Sau đó chia cho tổng số lượng cặp cùng nhãn để lấy trung bình
#         pos_per_row = ground_truth.sum(1)
#         local_loss = - (ground_truth * log_probs).sum(1) / pos_per_row
#         local_loss = local_loss.mean()

#         total_loss = local_loss + (self.vq_weight * vq_loss)
#         return total_loss, local_loss



class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        attn_weights = self.attention(x) 
        
        attn_weights = F.softmax(attn_weights, dim=1)
        
        global_reps = torch.sum(x * attn_weights, dim=1) 
        return global_reps
    
class ECGClusterAndClassifier(nn.Module):
    def __init__(self, backbone, embed_dim=256, mlp_hidden_dim=128, num_classes=2):
        super().__init__()
        self.backbone = backbone
        
        self.attn_pool = AttentionPooling(embed_dim)
        
        self.cluster_bottleneck = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU()
        )
        self.mlp_classifier = nn.Linear(mlp_hidden_dim, num_classes)

    def forward(self, source, padding_mask=None):
        backbone_out = self.backbone(source, padding_mask=padding_mask)
        local_reps = backbone_out["local_reps"] 

    
        global_reps = self.attn_pool(local_reps)

        feat = self.cluster_bottleneck(global_reps)
        logits = self.mlp_classifier(feat)
        
        return logits, feat
    

class ECGClusterClassifier:
    def __init__(self, n_neighbors=5, epsilon=1e-5):
        self.cluster_dictionary = {}
        self.clusterer = HDBSCAN(min_cluster_size=15, metric='euclidean')
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon
        self.class_weights = {0: 1.0, 1: 1.0}
        self.global_train_mean = None 

    def fit(self, train_features, train_labels):
        
       
        self.global_train_mean = np.mean(train_features, axis=0)

        centered_features = train_features - self.global_train_mean
        
        cluster_ids = self.clusterer.fit_predict(centered_features)
        
        unique_clusters = set(cluster_ids)
        if -1 in unique_clusters:
            unique_clusters.remove(-1) 
            
        af_cluster_count = 0
        non_af_cluster_count = 0
            
        for cid in unique_clusters:
            idx = np.where(cluster_ids == cid)[0]
            cluster_feats = centered_features[idx]
            cluster_lbls = train_labels[idx]
            
            centroid = np.mean(cluster_feats, axis=0)
            
            labels, counts = np.unique(cluster_lbls, return_counts=True)
            majority_label = labels[np.argmax(counts)]
            
            if majority_label == 1:
                af_cluster_count += 1
            else:
                non_af_cluster_count += 1
            
            self.cluster_dictionary[cid] = {
                'centroid': centroid,
                'label': majority_label,
                'size': len(idx)
            }
            
        print(f"DICTIONARY {len(self.cluster_dictionary)}  (AF: {af_cluster_count}, Non-AF: {non_af_cluster_count})")
        
        total = af_cluster_count + non_af_cluster_count
        if af_cluster_count > 0 and non_af_cluster_count > 0:
            self.class_weights[0] = total / (2.0 * non_af_cluster_count)
            self.class_weights[1] = total / (2.0 * af_cluster_count)

    def predict(self, test_features):
        if self.global_train_mean is None:
            raise ValueError("MODEL NOT fit! CALL FUNCTION fit BEFORE predict.")

      
        centered_test = test_features - self.global_train_mean

        predictions = []
        all_centroids = np.array([v['centroid'] for v in self.cluster_dictionary.values()])
        all_labels = np.array([v['label'] for v in self.cluster_dictionary.values()])
        
        distances = euclidean_distances(centered_test, all_centroids)
        
        for dist_row in distances:
            nearest_idx = np.argsort(dist_row)[:self.n_neighbors]
            nearest_distances = dist_row[nearest_idx]
            nearest_labels = all_labels[nearest_idx]
            
            weights = 1.0 / (nearest_distances + self.epsilon)
            
            label_scores = {0: 0.0, 1: 0.0}
            for label, weight in zip(nearest_labels, weights):
                label_scores[label] += weight * self.class_weights[label]
                    
            final_label = max(label_scores, key=label_scores.get)
            predictions.append(final_label)
            
        return np.array(predictions)

if __name__ == "__main__":
    batch = 1
    channel = 1
    length = 2400
    ecg_dum = torch.randn(batch, channel, length)
    model = ECGTransformerModel()
    result = model(ecg_dum)
    print("ecg dummy: ", ecg_dum)
