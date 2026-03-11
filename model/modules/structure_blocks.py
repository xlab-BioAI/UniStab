import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import StructureAwareFusion

class StructureAwareEncoder(nn.Module):
    """
    multi-layer structure aware encoder
    """
    def __init__(
        self,
        seq_dim=1024,
        pair_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_blocks=3,  
        dropout=0.1
    ):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([
            StructureAwareFusion(
                seq_dim=seq_dim,
                pair_dim=pair_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        
    def forward(self, seq_feats, pair_feats, mask):
        """
        Args:
            seq_feats: [B, L, seq_dim]
            pair_feats: [B, L, L, pair_dim]  
            mask: [B, L]
        Returns:
            enhanced_feats: [B, L, seq_dim]
        """
        x = seq_feats
        
        for i, block in enumerate(self.blocks):
            x = block(x, pair_feats, mask)
            
            # optional: add dropout to prevent overfitting
            if self.training and i < len(self.blocks) - 1:      
                x = F.dropout(x, p=0.1, training=self.training)
        
        return x