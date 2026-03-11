import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class StructureAwareFusion(nn.Module):
    def __init__(
        self, 
        seq_dim=1024,
        pair_dim=128,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.seq_dim = seq_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # sequence feature projection
        self.seq_proj_q = nn.Linear(seq_dim, hidden_dim * num_heads)
        self.seq_proj_k = nn.Linear(seq_dim, hidden_dim * num_heads)
        self.seq_proj_v = nn.Linear(seq_dim, hidden_dim * num_heads)
        
        # pair feature projection
        self.pair_proj = nn.Linear(pair_dim, num_heads)
        
        # output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_heads + pair_dim, seq_dim),
            nn.LayerNorm(seq_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.layer_norm = nn.LayerNorm(seq_dim)
        
        # add weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """improved weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)  # small gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, seq_feats, pair_feats, mask):
        B, L = seq_feats.shape[:2]
        
        # check input features contain NaN
        if torch.isnan(seq_feats).any() or torch.isnan(pair_feats).any():
            return seq_feats  # directly return original features
        
        # multi-head attention
        Q = self.seq_proj_q(seq_feats).view(B, L, self.num_heads, -1)
        K = self.seq_proj_k(seq_feats).view(B, L, self.num_heads, -1)  
        V = self.seq_proj_v(seq_feats).view(B, L, self.num_heads, -1)
        
        # attention score: sequence + pair
        seq_att = torch.einsum('bihd,bjhd->bijh', Q, K) / np.sqrt(self.hidden_dim)
        pair_att = self.pair_proj(pair_feats)
        
        # merge attention - add numerical stability
        attention_logits = seq_att + pair_att
        
        # clip extreme values to prevent softmax explosion
        attention_logits = torch.clamp(attention_logits, min=-10, max=10)
        
        # apply mask
        if mask is not None:
            mask_2d = mask.unsqueeze(1) * mask.unsqueeze(2)
            attention_logits = attention_logits.masked_fill(
                ~mask_2d.unsqueeze(-1), -1e4  
            )
        
        # attention weights
        attention_weights = F.softmax(attention_logits, dim=2)
        
        # check attention weights contain NaN
        if torch.isnan(attention_weights).any():
            return seq_feats
        
        # feature aggregation
        seq_context = torch.einsum('bijh,bjhd->bihd', attention_weights, V)
        seq_context = seq_context.reshape(B, L, -1)
        
        # pair feature aggregation
        attention_avg = attention_weights.mean(-1)
        pair_context = torch.einsum('bij,bijp->bip', attention_avg, pair_feats)
        
        # feature fusion
        fused_feats = torch.cat([seq_context, pair_context], dim=-1)
        enhanced_feats = self.out_proj(fused_feats)
        
        # check output features contain NaN
        if torch.isnan(enhanced_feats).any():
            return seq_feats
        
        # residual connection
        result = self.layer_norm(seq_feats + enhanced_feats)
        
        # final check
        if torch.isnan(result).any():
            return seq_feats
            
        return result