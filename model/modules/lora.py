import torch
import torch.nn as nn
import math
from typing import Optional, List

class LoRAConfig:
    """LoRA config"""
    def __init__(
        self,
        enabled: bool = True,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
    ):
        self.enabled = enabled
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["linear_q"]

class EfficientLoRALinear(nn.Module):
    """
    original layer completely detach
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        original_weight: torch.Tensor = None,
        original_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # store original weight (completely detach)
        self.register_buffer('original_weight', original_weight.detach().clone())
        if original_bias is not None:
            self.register_buffer('original_bias', original_bias.detach().clone())
        else:
            self.register_buffer('original_bias', None)
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, dtype=original_weight.dtype, device=original_weight.device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=original_weight.dtype, device=original_weight.device))
        
        # initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        with torch.no_grad():
            original_out = torch.nn.functional.linear(x, self.original_weight, self.original_bias)
        
        # only this part participate in gradient calculation
        x_down = torch.nn.functional.linear(x, self.lora_A.T)  # [..., rank]
        x_down = self.dropout(x_down)
        lora_out = torch.nn.functional.linear(x_down, self.lora_B.T)  # [..., out_features]
        
        # original_out.detach() ensure no gradient propagation
        return original_out + lora_out * self.scaling


def replace_linear_with_lora(
    module: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    prefix: str = ""
) -> int:
    """
    recursively replace the specified Linear layer with LoRA layer
    """
    replaced_count = 0
    
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        
        if isinstance(child, nn.Linear):
            # check if match target module
            should_replace = any(target in full_name for target in target_modules)
            
            if should_replace:
                # print original layer info for debugging
                print(f"🔧 original layer {full_name}: {child.in_features} -> {child.out_features}")
                
                # create efficient LoRA layer
                lora_layer = EfficientLoRALinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    original_weight=child.weight,
                    original_bias=child.bias
                )
                
                setattr(module, name, lora_layer)
                replaced_count += 1
                print(f"✅ replace layer: {full_name} (LoRA parameters: {rank * (child.in_features + child.out_features):,})")
        else:
            # recursively process submodule
            replaced_count += replace_linear_with_lora(
                child, target_modules, rank, alpha, dropout, full_name
            )
    
    return replaced_count