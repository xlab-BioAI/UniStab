
import torch
import pickle
import pandas as pd
from ipdb import set_trace

class UniversalMutationDataset(torch.utils.data.Dataset):
    
    def __init__(self, csv_path):
        
        
        data_name = csv_path
        
        self.df = pd.read_csv(data_name, usecols=["WT_name","ddG_ML", "mut_seq","wt_seq"])
        
        print('Including %s mutations' % str(self.df.shape[0]))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        row = self.df.iloc[idx]
        
        return {
            "WT_name": row.WT_name,
            "mut_seq": row.mut_seq,
            "wt_seq": row.wt_seq,
            "ddG": float(row.ddG_ML)  
        }

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    csv_path = "/home/xy_th/double_mutation_project/unistab/data/ptmul_with_mutant_seq.csv"
    
    dataset = UniversalMutationDataset(csv_path)
    
    print(len(dataset))
    
    sample =dataset[0]
    
    set_trace()
    
    print(f"sample type: {type(sample)}")

        
        