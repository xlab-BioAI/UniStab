import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
from scipy.stats import pearsonr, spearmanr
from model.unistab import V2Model, V2ModelConfig

class LightningDDGModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = V2Model(config)
        self.criterion = nn.MSELoss()
        self.save_hyperparameters()
        # initialize list to store validation step outputs
        self.val_preds = None
        self.val_targets = None
        
    def forward(self, batch):

        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        pred_ddg = outputs['ddg']
        target_ddg = batch['ddg']  
        
        loss = self.criterion(pred_ddg, target_ddg.squeeze()) 
        
        pcc, rmse, spearman = self._compute_metrics(pred_ddg, target_ddg.squeeze())
        
        # record metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_pcc', pcc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_spearman', spearman, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        pred_ddg = outputs['ddg']
        target_ddg = batch['ddg']  
        
        loss = self.criterion(pred_ddg, target_ddg.squeeze())
        
    
        pcc, rmse, spearman = self._compute_metrics(pred_ddg, target_ddg.squeeze())
        
    
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_pcc', pcc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_spearman', spearman, on_step=False, on_epoch=True, prog_bar=True)
        
  
        return {'loss': loss, 'preds': pred_ddg, 'targets': target_ddg.squeeze()}
    
    def on_validation_epoch_start(self):

        self._val_preds = []
        self._val_targets = []
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):

        self._val_preds.append(outputs['preds'])
        self._val_targets.append(outputs['targets'])
    
    def on_validation_epoch_end(self):

        all_preds = torch.cat(self._val_preds)
        all_targets = torch.cat(self._val_targets)
        

        pcc, rmse, spearman = self._compute_metrics(all_preds, all_targets)
        
   
        self.log('val_pcc_epoch', pcc)
        self.log('val_rmse_epoch', rmse)
        self.log('val_spearman_epoch', spearman)

        self.val_preds = all_preds
        self.val_targets = all_targets
    
    def configure_optimizers(self):
      
        params = self.model.parameters()
        
        # create optimizer
        optimizer = torch.optim.AdamW(
            params, 
            lr=self.config.train.lr, 
            weight_decay=self.config.train.weight_decay
        )
        
        # learning rate scheduler (optional)
        if hasattr(self.config, 'use_scheduler') and self.config.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_pcc"
                }
            }
        
        return optimizer
    
    def _compute_metrics(self, pred, target):
        """compute PCC, RMSE and Spearman correlation coefficient"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        pcc, _ = pearsonr(pred_np, target_np)
        
        rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
        
     
        spearman, _ = spearmanr(pred_np, target_np)
        
        return pcc, rmse, spearman


