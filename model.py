"""
LandslideSegFormer: Lightweight Landslide Segmentation with Enhanced Mix Transformer

A lightweight semantic segmentation model designed for landslide detection,
combining an Enhanced Mix Transformer (EMiT) encoder with a Gated Hierarchical
Attention Decoder (GHAD).

Architecture Overview:
- EMiT Encoder: B0-level transformer with efficient self-attention
- Parallel CNN Branch: Local feature extraction for edge preservation  
- Non-Local Block: Global context modeling
- GHAD Decoder: Lightweight MLP decoder with hierarchical attention

Paper: [Your Paper Title]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


# =============================================================================
# Basic Building Blocks
# =============================================================================

class DsConv2d(nn.Module):
    """Depthwise Separable Convolution."""
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D feature maps."""
    
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
    
    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, unbiased=False, keepdim=True).sqrt()
        return (x - mean) / (std + self.eps) * self.weight + self.bias


# =============================================================================
# EMiT Encoder Components
# =============================================================================

class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with spatial reduction."""
    
    def __init__(self, dim, heads=8, reduction_ratio=4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Conv2d(dim, dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, 
                               stride=reduction_ratio, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=False)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = kv.chunk(2, dim=1)
        
        # Reshape for attention
        q = q.view(B, self.heads, self.head_dim, -1).permute(0, 1, 3, 2)
        k = k.view(B, self.heads, self.head_dim, -1)
        v = v.view(B, self.heads, self.head_dim, -1).permute(0, 1, 3, 2)
        
        # Attention
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.to_out(out)


class MixFFN(nn.Module):
    """Mix Feed-Forward Network with depthwise convolution."""
    
    def __init__(self, dim, expansion=4):
        super().__init__()
        hidden = dim * expansion
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        return self.fc2(x)


class TransformerBlock(nn.Module):
    """Transformer block with efficient attention and Mix-FFN."""
    
    def __init__(self, dim, heads, expansion, reduction_ratio):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = EfficientSelfAttention(dim, heads, reduction_ratio)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = MixFFN(dim, expansion)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class EMiTStage(nn.Module):
    """Single stage of Enhanced Mix Transformer."""
    
    def __init__(self, in_channels, out_channels, num_layers, heads, 
                 expansion, reduction_ratio, patch_size=3, stride=2):
        super().__init__()
        
        # Overlapping patch embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, patch_size, stride=stride, 
                     padding=patch_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(out_channels, heads, expansion, reduction_ratio)
            for _ in range(num_layers)
        ])
        
        # Local enhancement (key component of EMiT)
        self.local_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.patch_embed(x)
        residual = x
        
        for block in self.blocks:
            x = block(x)
        
        # Local enhancement with residual
        x = x + self.local_enhance(x) + residual
        return x


class EMiTEncoder(nn.Module):
    """Enhanced Mix Transformer Encoder (B0-level)."""
    
    def __init__(self, in_channels=3, dims=(32, 64, 160, 256), 
                 heads=(1, 2, 5, 8), num_layers=(2, 2, 2, 2),
                 expansions=(8, 8, 4, 4), reduction_ratios=(8, 4, 2, 1)):
        super().__init__()
        
        self.stages = nn.ModuleList()
        
        for i in range(4):
            in_ch = in_channels if i == 0 else dims[i-1]
            patch_size = 7 if i == 0 else 3
            stride = 4 if i == 0 else 2
            
            self.stages.append(EMiTStage(
                in_channels=in_ch,
                out_channels=dims[i],
                num_layers=num_layers[i],
                heads=heads[i],
                expansion=expansions[i],
                reduction_ratio=reduction_ratios[i],
                patch_size=patch_size,
                stride=stride
            ))
    
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# =============================================================================
# Parallel CNN Branch
# =============================================================================

class CNNBranch(nn.Module):
    """Parallel CNN branch for local feature extraction."""
    
    def __init__(self, in_channels=3, dims=(32, 64, 128, 256)):
        super().__init__()
        
        self.stages = nn.ModuleList()
        in_ch = in_channels
        
        for i, out_ch in enumerate(dims):
            if i == 0:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 7, stride=4, padding=3, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            else:
                self.stages.append(nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ))
            in_ch = out_ch
    
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


# =============================================================================
# Feature Fusion
# =============================================================================

class FeatureFusion(nn.Module):
    """Fuse features from Transformer and CNN branches."""
    
    def __init__(self, trans_dims, cnn_dims):
        super().__init__()
        
        self.fusions = nn.ModuleList()
        for t_dim, c_dim in zip(trans_dims, cnn_dims):
            self.fusions.append(nn.Sequential(
                nn.Conv2d(t_dim + c_dim, t_dim, 1, bias=False),
                nn.BatchNorm2d(t_dim),
                nn.ReLU(inplace=True)
            ))
    
    def forward(self, trans_feats, cnn_feats):
        fused = []
        for fusion, t_feat, c_feat in zip(self.fusions, trans_feats, cnn_feats):
            # Align spatial dimensions
            if c_feat.shape[2:] != t_feat.shape[2:]:
                c_feat = F.interpolate(c_feat, size=t_feat.shape[2:], 
                                       mode='bilinear', align_corners=False)
            x = torch.cat([t_feat, c_feat], dim=1)
            fused.append(fusion(x) + t_feat)
        return fused


# =============================================================================
# Non-Local Block
# =============================================================================

class NonLocalBlock(nn.Module):
    """Non-Local Block for global context modeling."""
    
    def __init__(self, in_channels, reduction=2):
        super().__init__()
        inter_channels = in_channels // reduction
        
        self.theta = nn.Conv2d(in_channels, inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, inter_channels, 1)
        self.g = nn.Conv2d(in_channels, inter_channels, 1)
        
        self.out = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        theta = self.theta(x).view(B, -1, H * W).permute(0, 2, 1)
        phi = self.phi(x).view(B, -1, H * W)
        g = self.g(x).view(B, -1, H * W).permute(0, 2, 1)
        
        attn = F.softmax(theta @ phi, dim=-1)
        out = (attn @ g).permute(0, 2, 1).view(B, -1, H, W)
        
        return x + self.out(out) * self.scale


# =============================================================================
# GHAD Decoder (Gated Hierarchical Attention Decoder)
# =============================================================================

class ChannelAttention(nn.Module):
    """Channel attention module."""
    
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)


class GHADDecoder(nn.Module):
    """Gated Hierarchical Attention Decoder."""
    
    def __init__(self, in_channels, embed_dim=256, num_classes=2):
        super().__init__()
        
        # Linear projections for each scale
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for ch in in_channels
        ])
        
        # Channel attention for each scale
        self.attns = nn.ModuleList([
            ChannelAttention(embed_dim) for _ in in_channels
        ])
        
        # Learnable scale weights (gating mechanism)
        self.gates = nn.Parameter(torch.ones(len(in_channels)))
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(embed_dim, num_classes, 1)
        )
    
    def forward(self, features):
        # Get target size from first feature
        H, W = features[0].shape[2] * 4, features[0].shape[3] * 4
        target_size = (H // 4, W // 4)
        
        # Process each scale
        outs = []
        gates = F.softmax(self.gates, dim=0)
        
        for i, (feat, proj, attn) in enumerate(zip(features, self.projs, self.attns)):
            x = proj(feat)
            x = attn(x)
            x = x * gates[i]
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        
        # Fuse and predict
        x = self.fusion(torch.cat(outs, dim=1))
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        return x


# =============================================================================
# Main Model
# =============================================================================

class LandslideSegFormer(nn.Module):
    """
    LandslideSegFormer: Lightweight Landslide Segmentation Model
    
    Architecture:
        - EMiT Encoder: Enhanced Mix Transformer (B0-level, ~3.7M params)
        - CNN Branch: Parallel local feature extraction
        - Non-Local Block: Global context modeling
        - GHAD Decoder: Gated Hierarchical Attention Decoder
    
    Args:
        num_classes: Number of output classes (default: 2 for binary segmentation)
        in_channels: Number of input channels (default: 3 for RGB)
        embed_dim: Decoder embedding dimension (default: 256)
    """
    
    def __init__(self, num_classes=2, in_channels=3, embed_dim=256):
        super().__init__()
        
        # Encoder dimensions (B0-level)
        self.dims = (32, 64, 160, 256)
        self.cnn_dims = (32, 64, 128, 256)
        
        # EMiT Encoder
        self.encoder = EMiTEncoder(
            in_channels=in_channels,
            dims=self.dims,
            heads=(1, 2, 5, 8),
            num_layers=(2, 2, 2, 2),
            expansions=(8, 8, 4, 4),
            reduction_ratios=(8, 4, 2, 1)
        )
        
        # Parallel CNN Branch
        self.cnn_branch = CNNBranch(in_channels, self.cnn_dims)
        
        # Feature Fusion
        self.fusion = FeatureFusion(self.dims, self.cnn_dims)
        
        # Non-Local Blocks (applied to middle stages)
        self.non_local = nn.ModuleList([
            nn.Identity(),  # Stage 1: skip
            NonLocalBlock(self.dims[1]),  # Stage 2
            NonLocalBlock(self.dims[2]),  # Stage 3
            NonLocalBlock(self.dims[3])   # Stage 4
        ])
        
        # GHAD Decoder
        self.decoder = GHADDecoder(self.dims, embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Dual-path feature extraction
        trans_feats = self.encoder(x)
        cnn_feats = self.cnn_branch(x)
        
        # Feature fusion
        fused_feats = self.fusion(trans_feats, cnn_feats)
        
        # Non-local enhancement
        enhanced_feats = [nl(feat) for nl, feat in zip(self.non_local, fused_feats)]
        
        # Decode
        out = self.decoder(enhanced_feats)
        
        return out


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    model = LandslideSegFormer(num_classes=2)
    model.eval()
    
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {params:.2f}M")
