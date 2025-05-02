import torch
from torch import nn
import torch.nn.functional as F
import math
from scipy.ndimage import label
import numpy as np
from .utils import normalize_attention, normalize_and_smooth_attention



class FusionNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, 1, 1),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch + 16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(16, out_ch, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        # return self.conv2(x + self.conv1(x))
        return self.conv2(torch.cat([x, self.conv1(x)], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(base_ch, out_ch, 3, 1, 1))

    def forward(self, in_tensor, condition=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        for encoded_feature, up_conv, up_sample in zip(reversed(encoded_features),
                                                       reversed(self.up_convs),
                                                       reversed(self.up_samples)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == "relu":
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor):
        return self.conv(in_tensor)


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor, encoded_feature):
        up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)


class EtaEncoder(nn.Module):
    def __init__(self, in_ch=1, out_ch=2):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 5, 1, 2),  # (*, 16, 224, 224)
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 64, 3, 1, 1),  # (*, 64, 224, 224)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1)
        )
        self.seq = nn.Sequential(
            nn.Conv2d(64 + in_ch, 32, 32, 32, 0),  # (*, 32, 7, 7)
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 7, 7, 0))

    # def forward(self, x):
    #     return self.seq(torch.cat([self.in_conv(x), x], dim=1))
    
    def forward(self, x):
        features = self.in_conv(x)                 # spatial features M_η
        eta = self.seq(torch.cat([features, x], dim=1))  # global latent
        return eta, features                       # <-- return both


class Patchifier(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 32, 32, 0),  # (*, in_ch, 224, 224) --> (*, 64, 7, 7)
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, out_ch, 1, 1, 0))

    def forward(self, x):
        return self.conv(x)


class ThetaEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 17, 9, 4),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),  # (*, 32, 28, 28)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1),  # (*, 64, 14, 14)
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1))  # (* 64, 7, 7)
        self.mean_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))

    def forward(self, x):
        features = self.conv(x)                  # spatial features M_θ
        mu = self.mean_conv(features)           # latent θ
        logvar = self.logvar_conv(features)
        return mu, logvar, features             # <-- now returns 3 outputs

# class ThetaEncoder(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 32, 32, 0),  # (*, in_ch, 224, 244) --> (*, 32, 7, 7)
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(32, 64, 1, 1, 0),
#             # nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1))
#         self.mu_conv = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(64, out_ch, 7, 7, 0))
#         self.logvar_conv = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(64, out_ch, 7, 7, 0))
#
#     def forward(self, x, patch_shuffle=False):
#         m = self.conv(x)
#         if patch_shuffle:
#             batch_size = m.shape[0]
#             num_features = m.shape[1]
#             num_patches_per_dim = m.shape[-1]
#             m = m.view(batch_size, num_features, -1)[:, :, torch.randperm(num_patches_per_dim ** 2)]
#             m = m.view(batch_size, num_features, num_patches_per_dim, num_patches_per_dim)
#         mu = self.mu_conv(m)
#         logvar = self.logvar_conv(m)
#         return mu, logvar


class AttentionModule(nn.Module):
    def __init__(self, dim, v_ch=5):
        super().__init__()
        self.dim = dim
        self.v_ch = v_ch
        self.q_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))
        self.k_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))

        self.scale = self.dim ** (-0.5)
        
    def forward(self, q, k, v, mask, modality_dropout=None, temperature=10.0):
        """
        Attention module for optimal anatomy fusion.

        ===INPUTS===
        * q: torch.Tensor (batch_size, feature_dim_q, num_q_patches=1)
            Query variable. In HACA3, query is the concatenation of target \theta and target \eta.
        * k: torch.Tensor (batch_size, feature_dim_k, num_k_patches=1, num_contrasts=4)
            Key variable. In HACA3, keys are \theta and \eta's of source images.
        * v: torch.Tensor (batch_size, self.v_ch=5, num_v_patches=224*224, num_contrasts=4)
            Value variable. In HACA3, values are multi-channel logits of source images.
            self.v_ch is the number of \beta channels.
        * modality_dropout: torch.Tensor (batch_size, num_contrasts=4)
            Indicates which contrast indexes have been dropped out. 1: if dropped out, 0: if exists.
        """
        batch_size, feature_dim_q, num_q_patches = q.shape
        _, feature_dim_k, _, num_contrasts = k.shape
        num_v_patches = v.shape[2]
        assert (
                feature_dim_k == feature_dim_q or feature_dim_q == self.feature_dim
        ), 'Feature dimensions do not match.'

        # q.shape: (batch_size, num_q_patches=1, 1, feature_dim_q)
        q = q.reshape(batch_size, feature_dim_q, num_q_patches, 1).permute(0, 2, 3, 1)
        # k.shape: (batch_size, num_k_patches=1, num_contrasts=4, feature_dim_k)
        k = k.permute(0, 2, 3, 1)
        # v.shape: (batch_size, num_v_patches=224*224, num_contrasts=4, v_ch=5)
        v = v.permute(0, 2, 3, 1)
        q = self.q_fc(q)
        # k.shape: (batch_size, num_k_patches=1, feature_dim_k, num_contrasts=4)
        k = self.k_fc(k).permute(0, 1, 3, 2)

        # dot_prod.shape: (batch_size, num_q_patches=1, 1, num_contrasts=4)
        dot_prod = (q @ k) * self.scale
        interpolation_factor = int(math.sqrt(num_v_patches // num_q_patches))

        q_spatial_dim = int(math.sqrt(num_q_patches))
        dot_prod = dot_prod.view(batch_size, q_spatial_dim, q_spatial_dim, num_contrasts)

        image_dim = int(math.sqrt(num_v_patches))
        # dot_prod_interp.shape: (batch_size, image_dim, image_dim, num_contrasts)
        dot_prod_interp = dot_prod.repeat(1, interpolation_factor, interpolation_factor, 1)
        if modality_dropout is not None:
            modality_dropout = modality_dropout.view(batch_size, num_contrasts, 1, 1).permute(0, 2, 3, 1)
            dot_prod_interp = dot_prod_interp - (modality_dropout.repeat(1, image_dim, image_dim, 1).detach() * 1e5)

        attention = (dot_prod_interp / temperature).softmax(dim=-1)
        # v = attention.view(batch_size, num_v_patches, 1, num_contrasts) @ v
        # v = v.view(batch_size, image_dim, image_dim, self.v_ch).permute(0, 3, 1, 2)
        # attention = attention.view(batch_size, image_dim, image_dim, num_contrasts).permute(0, 3, 1, 2)

        #print(f"Mask type: {mask.dtype}, shape: {mask.shape}")

        if isinstance(mask, list):
            mask = torch.stack(mask)

        #print(mask.shape)

        # Transpose the mask to match the order of dimensions in attention
        if len(mask.shape)==5 and mask.shape[0]!= batch_size:
            mask = mask.permute(1, 2, 3, 4, 0)  # This changes the order to [batch_size, num_contrasts, 1, 224, 224]
            #mask = mask.squeeze(1)  # Squeeze the -- dimension to reduce the shape to [56, 224, 224, 3]
        # print(mask.shape)
        mask = mask.squeeze(1)

        #print(f"Attention type: {attention.dtype}, shape: {attention.shape}")
        #print(f"Mask type: {mask.dtype}, shape: {mask.shape}")

        attention_map = attention * mask
        # print(f"Attention Map type: {attention_map.dtype}, shape: {attention_map.shape}")

        # Normalize the attention map
        # normalized_attention_map = normalize_attention(attention_map)
        normalized_attention_map = normalize_and_smooth_attention(attention_map,diff_threshold=0.2)
        # print(f"Normalization Attention Map type: {normalized_attention_map.dtype}, shape: {normalized_attention_map.shape}")

        # # Manual Attention Map (size:[56, 224, 224, 3])
        # normalized_attention_map[:, :112, :, 0] = 1  
        # normalized_attention_map[:, :112, :, 0] = 0 
        # normalized_attention_map[:, :112, :, 1] = 0 
        # normalized_attention_map[:, :112, :, 1] = 0.5
        # normalized_attention_map[:, :112, :, 2] = 0 
        # normalized_attention_map[:, :112, :, 2] = 0.5
        # # print(f"normalized_attention_map shape after manual modification: {normalized_attention_map.shape}")

        
        # Use the normalized attention map instead of the original attention for v calculation
        v = normalized_attention_map.view(batch_size, num_v_patches, 1, num_contrasts) @ v
        v = v.view(batch_size, image_dim, image_dim, self.v_ch).permute(0, 3, 1, 2)
        attention = normalized_attention_map.view(batch_size, image_dim, image_dim, num_contrasts).permute(0, 3, 1, 2)
        # print(f"[DEBUG] attention_weights shape: {attention.shape}")

        return v, attention

class SpatialAttentionModule(nn.Module):
    def __init__(self, feature_dim, key_dim=64, v_ch=8):
        """
        Spatial attention module using query/key/value mechanism.
        Args:
            feature_dim (int): input feature dim (e.g., theta+eta channels)
            key_dim (int): dimension to project query/key into
            v_ch (int): number of channels in the value maps (beta channels)
        """
        super(SpatialAttentionModule, self).__init__()
        self.query_net = nn.Conv2d(feature_dim, key_dim, kernel_size=1)
        self.key_net = nn.Conv2d(feature_dim, key_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.v_ch = v_ch

    def forward(self, q_feats, k_feats_list, beta_list, mask, modality_dropout=None, return_attention=False):
        """
        Args:
            q_feats: (B, C, H, W)
            k_feats_list: list of (B, C, H, W)
            beta_list: list of (B, beta_dim, H, W)
            mask: 
            modality_dropout: (B, N) with 1 = dropped out, 0 = keep

        Returns:
            beta_fused: (B, beta_dim, H, W)
            attention_weights: (B, N, H, W)
        """
        B, _, H, W = q_feats.shape
        N = len(k_feats_list)

        # Project q and k
        q_proj = self.query_net(q_feats)  # (B, key_dim, H, W)
        k_proj_list = [self.key_net(k) for k in k_feats_list]  # list of (B, key_dim, H, W)

        # Compute attention scores
        attention_scores = []
        for k in k_proj_list:
            attn = torch.sum(q_proj * k, dim=1, keepdim=True)  # (B, 1, H, W)
            attention_scores.append(attn)

        # Stack and normalize
        attention_stack = torch.cat(attention_scores, dim=1)  # (B, N, H, W)
        
        # === Apply modality dropout ===
        if modality_dropout is not None:
            # modality_dropout shape: [B, N] → [B, N, 1, 1] for broadcasting
            modality_dropout_mask = modality_dropout.view(B, N, 1, 1)
            attention_stack = attention_stack - (modality_dropout_mask * 1e5)

        attention_weights = self.softmax(attention_stack)     # (B, N, H, W)
        # print(f"[DEBUG] spatial attention_weights shape: {attention_weights.shape}")

        # === Apply brain region mask ===
        if isinstance(mask, list):
            mask = torch.stack(mask)
        
        # print("=== DEBUG MASK ===")
        # print(f"mask shape after stack: {mask.shape}")
        # print(f"mask dtype: {mask.dtype}")
        # print("===================")
        # mask = mask.permute(1, 0, 2, 3, 4).squeeze(2)
        mask = mask.permute(0, 4, 1, 2, 3).squeeze(2)
        # print("=== POST-PROCESS MASK ===")
        # print(f"mask shape after permute + squeeze: {mask.shape}")  # should be [B, N, H, W]

        upsampled_attention = F.interpolate(attention_weights, size=beta_list[0].shape[-2:], mode='bilinear', align_corners=False)
        # print(f"spatial upsampled_attention shape: {upsampled_attention.shape}")
        masked_attention = upsampled_attention # * mask
        
        masked_attention_perm = masked_attention.permute(0, 2, 3, 1)  # [B, H, W, N]
        normalized = normalize_attention(masked_attention_perm)
        normalize_attention_weights = normalized.permute(0, 3, 1, 2)  # Back to [B, N, H, W]

        # Example manual attention for testing purposes
        manual_attention = torch.zeros_like(normalize_attention_weights)
        
        # Set attention to be deterministic, e.g., modality 0 gets full attention at top half,
        # modality 1 gets full attention at bottom-left quadrant,
        # modality 2 gets full attention at bottom-right quadrant.
       # Set attention to be deterministic
        H, W = manual_attention.shape[2], manual_attention.shape[3]
        
        manual_attention[:, 0, :H//2, :] = 1.0          # top half
        manual_attention[:, 1, H//2:, :W//2] = 1.0      # bottom-left
        manual_attention[:, 2, H//2:, W//2:] = 1.0      # bottom-right
        
        # Normalize manually
        manual_attention_sum = manual_attention.sum(dim=1, keepdim=True) + 1e-8
        normalize_attention_weights = manual_attention / manual_attention_sum
        
        # Verify sum to 1 across modalities
        assert torch.allclose(normalize_attention_weights.sum(dim=1), torch.ones((B, H, W)))


        # Weighted fusion of beta
        beta_fused = torch.zeros_like(beta_list[0])
        for i in range(N):
            w = normalize_attention_weights[:, i:i+1, :, :]  # (B, 1, H, W)
            # if w.shape[-2:] != beta_list[i].shape[-2:]:
            #     w = F.interpolate(w, size=beta_list[i].shape[-2:], mode='bilinear', align_corners=False)
            beta_fused += w * beta_list[i]  # Broadcasted
        # print(f"[DEBUG] beta_fused shape: {beta_fused.shape}")
        if return_attention:
            # upsampled_attention = F.interpolate(attention_weights, size=beta_list[0].shape[-2:], mode='bilinear', align_corners=False)
            # print(f"[DEBUG] spatial upsampled_attention shape: {upsampled_attention.shape}")
            return beta_fused, normalize_attention_weights
        else:
            return beta_fused

