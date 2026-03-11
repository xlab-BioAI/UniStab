import pytorch_lightning as pl
from datasets import MegaDataset
from torch.utils.data import DataLoader
from utils.pdb_utils import tied_featurize_mut

class DDGDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.num_workers = cfg.train.num_workers
        
    def setup(self, stage=None):
        # load dataset
        self.train_dataset = MegaDataset(self.cfg, 'train')
        self.val_dataset = MegaDataset(self.cfg, 'val')
        if 'test' in self.cfg.data.splits:
            self.test_dataset = MegaDataset(self.cfg, 'test')
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda b: tied_featurize_mut(b)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: tied_featurize_mut(b)
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: tied_featurize_mut(b),
            pin_memory=True
        )
        
