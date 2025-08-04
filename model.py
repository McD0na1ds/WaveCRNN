import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import math

class LightweightViT(nn.Module):
    """Lightweight Vision Transformer for student model"""
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=384, depth=6, num_heads=6, 
                 mlp_ratio=4., num_classes=3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # DINOv2 uses 16x16 patches for 224x224 images, resulting in 256 patches + 1 cls token = 257
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # B, C, H, W -> B, embed_dim, H//patch_size, W//patch_size
        x = x.flatten(2).transpose(1, 2)  # B, embed_dim, N -> B, N, embed_dim
        
        # Add cls token and positional embedding
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x  # Return all tokens including cls token
    
    def forward(self, x):
        x = self.forward_features(x)
        # Use cls token for classification
        cls_token = x[:, 0]
        return self.head(cls_token), x[:, 1:]  # Return classification and patch tokens


class LSTMClassifier(nn.Module):
    """LSTM module for temporal analysis of patch features"""
    def __init__(self, input_dim=384, hidden_dim=256, num_layers=2, num_classes=3, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # Apply attention to focus on important features
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_dim*2)
        
        # Classification
        output = self.classifier(pooled)
        return output


class FeatureAdapter(nn.Module):
    """Adapter to map student features to teacher feature space"""
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.adapter = nn.Linear(student_dim, teacher_dim)
        
    def forward(self, x):
        return self.adapter(x)


class StudentModel(nn.Module):
    """Complete student model combining lightweight ViT and LSTM"""
    def __init__(self, num_classes=3):
        super().__init__()
        self.vit = LightweightViT(num_classes=num_classes)
        self.lstm = LSTMClassifier(num_classes=num_classes)
        
    def forward(self, x):
        # Get classification output and patch features from ViT
        cls_output, patch_features = self.vit(x)
        
        # Process patch features with LSTM
        lstm_output = self.lstm(patch_features)
        
        # Combine outputs (can be adjusted based on needs)
        combined_output = cls_output + lstm_output
        
        return combined_output, patch_features


def get_dinov2_model(model_name='dinov2_vits14'):
    """Load pre-trained DINOv2 model"""
    try:
        import torch.hub
        model = torch.hub.load('facebookresearch/dinov2', model_name)
        return model
    except Exception as e:
        print(f"Failed to load DINOv2 model: {e}")
        return None