import torch
import numpy as np
import torch.nn as nn
from ipdb import set_trace
from dataclasses import dataclass
from model.modules.pooling import *
from model.modules.esmfold_trunk import SimpleTrunkExtractor
from model.modules.lora import LoRAConfig
from model.modules.structure_blocks import StructureAwareEncoder


@dataclass
class V2ModelConfig:
    esm_name: str = 'esm2_t12_35M_UR50D'
    esm_tune: bool = False
    pooling_method: str = 'GlobalMaskValueAttentionPooling1D'
    
    # use trunk or not
    use_trunk: bool = True
    
    lora: LoRAConfig = True
    
    # structure encoder config
    structure_encoder: dict = None

class V2Model(nn.Module):
    _default_config = V2ModelConfig()

    def __init__(self, config: V2ModelConfig = _default_config):
        super(V2Model, self).__init__()
        self.config = config
        
        
        self.feature_extractor = SimpleTrunkExtractor(lora_config=config.lora)
        self.feature_dim = 1024
        
        
        structure_config = config.structure_encoder or {}
        self.structure_encoder = StructureAwareEncoder(
            seq_dim=1024,
            pair_dim=128,
            hidden_dim=structure_config.get('hidden_dim', 256),
            num_heads=structure_config.get('num_heads', 8),
            num_blocks=structure_config.get('num_blocks', 3),
            dropout=structure_config.get('dropout', 0.1)
        )
        
      
        self.pooling_layer = self._get_pooling_layer(config.pooling_method, self.feature_dim)

        
        self.ddg_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _get_pooling_layer(self, pooling_method, embed_size):
        pooling_map = {
            'GlobalMaskMaxPooling1D': GlobalMaskMaxPooling1D(),
            'GlobalMaskMinPooling1D': GlobalMaskMinPooling1D(),
            'GlobalMaskAvgPooling1D': GlobalMaskAvgPooling1D(),
            'GlobalMaskSumPooling1D': GlobalMaskSumPooling1D(axis=1),
            'GlobalMaskWeightedAttentionPooling1D': GlobalMaskWeightedAttentionPooling1D(embed_size),
            'GlobalMaskContextAttentionPooling1D': GlobalMaskContextAttentionPooling1D(embed_size),
            'GlobalMaskValueAttentionPooling1D': GlobalMaskValueAttentionPooling1D(embed_size),
            'GlobalMaxPool1d': GlobalMaxPool1d(),
            'GlobalAvgPool1d': GlobalAvgPool1d(),
            'AttentionPool1d': AttentionPool1d(embed_size),
        }
        return pooling_map[pooling_method]

    def forward(self, batch):
        wt_tokens = batch['wt_tokens']
        mut_tokens = batch['mut_tokens']
        wt_mask = batch['wt_mask']
        mut_mask = batch['mut_mask']
        
        # ESMFold feature extraction (return sequence and pair features)
        wt_seq_feats, wt_pair_feats = self.feature_extractor(wt_tokens, wt_mask)
        mut_seq_feats, mut_pair_feats = self.feature_extractor(mut_tokens, mut_mask)
        
        # structure aware feature enhancement
        wt_enhanced = self.structure_encoder(wt_seq_feats, wt_pair_feats, wt_mask)
        mut_enhanced = self.structure_encoder(mut_seq_feats, mut_pair_feats, mut_mask)
        del  wt_seq_feats, wt_pair_feats,  mut_seq_feats, mut_pair_feats
      
        # pooling and prediction
        wt_pooled = self.pooling_layer(wt_enhanced)
     
        mut_pooled = self.pooling_layer(mut_enhanced)

        pooled_diff = mut_pooled - wt_pooled
        pred_ddg = self.ddg_head(pooled_diff).squeeze(-1)
        
        return {'ddg': pred_ddg}