import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
import argparse
from torch.utils.data import DataLoader

from lightning_model import LightningDDGModel
from datasets.dataset import UniversalMutationDataset
from utils.pdb_utils import tied_featurize_mut
from omegaconf import OmegaConf


def run_inference(checkpoint_path: str, csv_path: str, output_dir: str, 
                 dataset_name: str = "Test", 
                 batch_size: int = 32, device: str = None):
    """
    inference function
    
    Args:
        checkpoint_path: model checkpoint path
        csv_path: data csv path
        output_dir: output directory
        dataset_name: dataset name
        batch_size: batch size
        device: device name
    """
    # device selection
    if device is None:
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
    # configuration
    cfg = OmegaConf.create({
        "data": {"side_chains": False},
        "train": {"batch_size": batch_size, "num_workers": 2}
    })
    
    # load model
    print(f"load model: {checkpoint_path}")
    model = LightningDDGModel.load_from_checkpoint(checkpoint_path)
    model.eval().to(device)

    # enable gradient checkpointing to reduce memory usage
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # load dataset
    dataset = UniversalMutationDataset(csv_path)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: tied_featurize_mut(b)
    )
    
    # inference
    all_preds, all_targets = [], []
    all_esm_ddg, all_mpnn_ddg = [], []
    
    print("start inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"process batch {batch_idx}/{len(dataloader)}")
            
            if batch is None:
                continue
                
            # move data to device
            for key in ['wt_tokens', 'mut_tokens', 'wt_mask', 'mut_mask', 'ddg']:
                if key in batch:
                    batch[key] = batch[key].to(device)
            
            try:
                outputs = model(batch)
                all_preds.append(outputs['ddg'].detach().cpu())
                all_targets.append(batch['ddg'].detach().cpu())
                
                if 'esm_ddg' in outputs:
                    all_esm_ddg.append(outputs['esm_ddg'].detach().cpu())
                if 'mpnn_ddg' in outputs:
                    all_mpnn_ddg.append(outputs['mpnn_ddg'].detach().cpu())
            except Exception as e:
                print(f"batch {batch_idx} failed: {e}")
                continue
    
    # merge results
    preds = torch.cat(all_preds).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()
    
    # remove invalid values
    valid_mask = np.isfinite(preds) & np.isfinite(targets)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    print(f"valid samples: {len(preds)}")
    
    # calculate metrics
    def calc_metrics(pred, target, name):
        if len(pred) == 0 or np.std(pred) == 0:
            return {'pcc': np.nan, 'spearman': np.nan, 'rmse': np.nan, 'mae': np.nan}
        
        pcc, _ = pearsonr(pred, target)
        spearman, _ = spearmanr(pred, target)
        rmse = np.sqrt(np.mean((pred - target) ** 2))
        mae = np.mean(np.abs(pred - target))
        
        print(f"{name}: PCC={pcc:.4f}, Spearman={spearman:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        return {'pcc': pcc, 'spearman': spearman, 'rmse': rmse, 'mae': mae}
    
    # evaluaton
    metrics = calc_metrics(preds, targets, "UniStab")
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # plot scatter
    def plot_scatter(pred, target, title, save_path, metrics):
        if len(pred) == 0 or np.isnan(metrics['pcc']):
            return
            
        plt.figure(figsize=(8, 8))
        plt.scatter(target, pred, alpha=0.6, s=20)
        plt.xlabel('Experimental ΔΔG')
        plt.ylabel('Predicted ΔΔG')
        plt.title(f'{title}\nPCC: {metrics["pcc"]:.3f}, RMSE: {metrics["rmse"]:.3f}')
        
        # diagonal
        min_val, max_val = min(pred.min(), target.min()), max(pred.max(), target.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"save figure: {save_path}")
    
    # save figure
    plot_scatter(preds, targets, f"{dataset_name} - UniStab", 
                os.path.join(output_dir, f"{dataset_name}_fusion.png"), metrics)
    
    # save results
    results = {
        'predictions': preds,
        'targets': targets,
        'fusion_metrics': metrics
    }
    
    
    np.savez(os.path.join(output_dir, f"{dataset_name}_results.npz"), **results)
    
    # save checkpoint info
    with open(os.path.join(output_dir, "checkpoint.txt"), "w") as f:
        f.write(f"{checkpoint_path}\n")
    
    print(f"results saved to: {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='data csv path')
    parser.add_argument('--output', type=str, default='./results', help='output directory')
    parser.add_argument('--name', type=str, default='Test', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--device', type=str, default=None, help='device')
    
    args = parser.parse_args()
    
    run_inference(
        checkpoint_path=args.checkpoint,
        csv_path=args.data,
        output_dir=args.output,
        dataset_name=args.name,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()