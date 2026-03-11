# https://github.com/BytedProtein/ByProt/blob/dd279dc85f76ee2c28c819b71bf3911b90159f0a/src/byprot/utils/config.py
import os
import yaml
import logging
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def load_config(config_path: str, overrides: list = None) -> DictConfig:
    """add config file and apply overrides"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    
    # load base config
    cfg = OmegaConf.load(config_path)
    
    # parse environment variables
    cfg = OmegaConf.create(OmegaConf.to_yaml(cfg, resolve=True))
    
    # apply command line overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    return cfg


def save_config(cfg: DictConfig, save_path: str):
    """save config to file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(OmegaConf.to_yaml(cfg), f, default_flow_style=False)

def compose_config(**kwds):
    return OmegaConf.create(kwds)


def merge_config(config, override_config):
    return OmegaConf.merge(config, override_config)


