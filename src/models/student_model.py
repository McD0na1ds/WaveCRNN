import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_height, patch_width = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, return_features=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        if self.pool == 'mean':
            features = x[:, 1:].mean(dim=1)  # Exclude cls token
        else:
            features = x[:, 0]  # Use cls token

        features = self.to_latent(features)
        
        if return_features:
            return features
        
        return self.mlp_head(features)


class StudentCRNN(nn.Module):
    def __init__(self, vit_config, lstm_config, num_classes):
        super().__init__()
        
        # ViT backbone
        self.vit = ViT(**vit_config)
        vit_dim = vit_config['dim']
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=vit_dim,
            hidden_size=lstm_config['hidden_size'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'] if lstm_config['num_layers'] > 1 else 0,
            bidirectional=lstm_config['bidirectional'],
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = lstm_config['hidden_size']
        if lstm_config['bidirectional']:
            lstm_output_dim *= 2
            
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_output_dim),
            nn.Dropout(0.1),
            nn.Linear(lstm_output_dim, num_classes)
        )
        
        # Feature extraction layer for knowledge distillation
        self.feature_proj = nn.Linear(lstm_output_dim, 768)  # Project to DINOv2 feature dimension
        
    def forward(self, x, return_features=False):
        # Get features from ViT
        vit_features = self.vit(x, return_features=True)  # [B, dim]
        
        # Add sequence dimension for LSTM (treating each image as a single time step)
        lstm_input = vit_features.unsqueeze(1)  # [B, 1, dim]
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        # Use the last hidden state
        if self.lstm.bidirectional:
            # Concatenate forward and backward hidden states
            features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            features = hidden[-1]
        
        if return_features:
            # Project features to match teacher dimension for distillation
            projected_features = self.feature_proj(features)
            return projected_features
            
        # Classification
        logits = self.classifier(features)
        return logits
    
    def get_features_and_logits(self, x):
        """获取特征和logits，用于知识蒸馏"""
        vit_features = self.vit(x, return_features=True)
        lstm_input = vit_features.unsqueeze(1)
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        if self.lstm.bidirectional:
            features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            features = hidden[-1]
        
        projected_features = self.feature_proj(features)
        logits = self.classifier(features)
        
        return projected_features, logits