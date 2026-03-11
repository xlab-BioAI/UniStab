
import torch
import pickle
import pandas as pd
from ipdb import set_trace

class MegaDataset(torch.utils.data.Dataset):
    
    def __init__(self, cfg, split):
        
        self.cfg = cfg
        self.split = split
        
        data_name = self.cfg.data_loc.megascale_csv
       
        
        self.df = pd.read_csv(data_name, usecols=["ddG_ML", "WT_name", "mut_seq","wt_seq"])
        
        # load splits produced by mmseqs clustering
        with open(self.cfg.data_loc.megascale_splits, 'rb') as f:
            splits = pickle.load(f)
        self.wt_names = splits[self.split]
        
        self.df = self.df[self.df['WT_name'].isin(self.wt_names)]
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
            "ddG": -1.0 * float(row.ddG_ML)  
        }

if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create(
        {
            "data_loc":{
                "megascale_csv": "/home/xy_th/double_mutation_project/unistab/data/all_data.csv",
                "megascale_splits": "/home/xy_th/double_mutation_project/unistab/data/splits.pkl"
            }
            
        }
        
    )
    
    dataset = MegaDataset(cfg, "train_ptmul")
    
    print(len(dataset))
    
    sample =dataset[0]
    
    set_trace()
    
    print(f"sample type: {type(sample)}")

        
        