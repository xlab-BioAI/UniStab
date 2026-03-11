import torch
from ipdb import set_trace
from torch.utils.data import DataLoader

def tied_featurize_mut(batch, device='cpu'):
    """use standard OpenFold encoding"""
    from openfold.np import residue_constants
    
    wt_seqs = [b['wt_seq'] for b in batch]
    mut_seqs = [b['mut_seq'] for b in batch]
    ddg_values = [[b['ddG']] for b in batch]
    
    def encode_sequences(sequences):
        encoded = []
        for seq in sequences:
            tokens = [residue_constants.restype_order_with_x.get(aa, 20) for aa in seq]
            encoded.append(tokens)
        
        max_len = max(len(tokens) for tokens in encoded) if encoded else 0
        batch_tokens = torch.zeros((len(sequences), max_len), dtype=torch.long)
        batch_mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
        
        for i, tokens in enumerate(encoded):
            if tokens:
                batch_tokens[i, :len(tokens)] = torch.tensor(tokens)
                batch_mask[i, :len(tokens)] = 1
        
        return batch_tokens, batch_mask
    
    wt_tokens, wt_mask = encode_sequences(wt_seqs)
    mut_tokens, mut_mask = encode_sequences(mut_seqs)
    
    return {
        'wt_tokens': wt_tokens.to(device),
        'mut_tokens': mut_tokens.to(device),
        'wt_mask': wt_mask.to(device),
        'mut_mask': mut_mask.to(device),
        'ddg': torch.tensor(ddg_values, dtype=torch.float32, device=device)
    }
if __name__ == "__main__":
    from datasets.megadataset import MegaDataset
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create(
        {
            "data_loc":{
                "megascale_csv": "data/mutation_data_with_wildtype.csv",
                "megascale_splits": "data/splits.pkl"
                
            }
        }
        
    )
    
    dataset = MegaDataset(cfg, "train")
    
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=lambda x: tied_featurize_mut(x))
    
    for batch in dataloader:
        print(batch)
        set_trace()
        print("test")
