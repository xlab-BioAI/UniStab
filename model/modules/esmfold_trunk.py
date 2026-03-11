import torch
import torch.nn as nn
import esm
from .lora import replace_linear_with_lora, LoRAConfig
from torch.utils.checkpoint import checkpoint

class SimpleTrunkExtractor(nn.Module):
    """
    simplified ESMFold trunk feature extractor, default use LoRA
    """
    
    def __init__(self, lora_config: LoRAConfig = None):
        super().__init__()
        # directly load pretrained ESMFold
        self.esmfold = esm.pretrained.esmfold_v1()
        self.esmfold.esm.half()

        # default apply LoRA
        if lora_config and lora_config.enabled:
            self._apply_lora(lora_config)
        else:
            # if no LoRA config, freeze all parameters
            for param in self.esmfold.parameters():
                param.requires_grad = False
                
    def _apply_lora(self, lora_config: LoRAConfig):
        """apply LoRA to ESMFold"""        
        # first freeze all parameters
        for param in self.esmfold.parameters():
            param.requires_grad = False
            
        # only apply LoRA to trunk.blocks
        replaced_count = replace_linear_with_lora(
            module=self.esmfold.trunk.blocks,
            target_modules=lora_config.target_modules,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout
        )
                
        # calculate trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"📊 trainable parameters: {trainable_params:,}")
        print(f"📊 total parameters: {total_params:,}")
        print(f"📊 trainable ratio: {100 * trainable_params / total_params:.4f}%")
        
    def forward(self, aa_tokens, mask):
        """
        simplified forward propagation: ESM part detach, trunk part may have LoRA gradient
        """
        B, L = aa_tokens.shape
        device = aa_tokens.device
        
        # ESM language model part: completely freeze
        with torch.no_grad():
            esmaa = self.esmfold._af2_idx_to_esm_idx(aa_tokens, mask)
            residx = torch.arange(L, device=device).expand(B, L)
            
            # ESM language model part
            esm_s = self.esmfold._compute_language_model_representations(esmaa)
            esm_s = esm_s.to(self.esmfold.esm_s_combine.dtype)
            esm_s = esm_s.detach()
            
            # preprocess
            mask_float = mask.float()
            esm_s = (self.esmfold.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
            
            # trunk input preparation
            s_s_0 = self.esmfold.esm_s_mlp(esm_s) + self.esmfold.embedding(aa_tokens)
            s_z_0 = s_s_0.new_zeros(B, L, L, self.esmfold.cfg.trunk.pairwise_state_dim)
            z = s_z_0 + self.esmfold.trunk.pairwise_positional_embedding(residx, mask=mask_float)
        
        s = s_s_0
        z_curr = z
        
        def checkpoint_block(block, s, z, mask, residx, chunk_size):
            return block(s, z, mask=mask, residue_index=residx, chunk_size=chunk_size)

        for block in self.esmfold.trunk.blocks:
            s, z_curr = checkpoint(checkpoint_block, block, s, z_curr, mask_float, residx, None)
        
        
        return s, z_curr