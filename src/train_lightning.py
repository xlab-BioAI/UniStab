import os
import argparse
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from lightning_model import LightningDDGModel
from lightning_data import DDGDataModule
from utils.config import load_config, save_config

    
def main():
    parser = argparse.ArgumentParser(description='Train DDG prediction model')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--overrides', type=str, nargs='*', default=[])
    args = parser.parse_args()
    
    # load config
    cfg = load_config(args.config, overrides=args.overrides)
        
    # create model
    from model.unistab import V2ModelConfig
    from model.modules.lora import LoRAConfig
    pl.seed_everything(42)
    # create LoRA config
    lora_config = None
    if hasattr(cfg.model, 'lora') and cfg.model.lora.enabled:
        lora_config = LoRAConfig(
            enabled=cfg.model.lora.enabled,
            rank=cfg.model.lora.rank,
            alpha=cfg.model.lora.alpha,
            dropout=cfg.model.lora.dropout,
            target_modules=cfg.model.lora.target_modules
        )
    
    model_config = V2ModelConfig(
        pooling_method=cfg.model.pooling_method,
        lora=lora_config,
        structure_encoder=cfg.model.structure_encoder
    )
    model_config.train = cfg.train
    
    # data and model
    dm = DDGDataModule(cfg)
    model = LightningDDGModel(model_config)
    
    # callbacks and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.save_dir,
        filename='best-{epoch:02d}-{val_pcc:.4f}',
        monitor='val_pcc',
        mode='max',
        save_top_k=1,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_pcc',
        patience=cfg.train.patience,
        mode='max'
    )
    
    logger = TensorBoardLogger(cfg.train.save_dir, name="", version="")
    
    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        strategy=cfg.hardware.strategy,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
        precision=cfg.hardware.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches
    )
    
    # train
    trainer.fit(model, dm)
    
    # results
    if trainer.is_global_zero:
        print(f"best model: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()