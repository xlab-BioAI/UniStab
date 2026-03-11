import torch
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import heapq
from dataclasses import dataclass
from collections import defaultdict
import itertools
from lightning_model import LightningDDGModel
from utils.pdb_utils import tied_featurize_mut
import copy

@dataclass
class OptimizationConfig:
    """optimization config"""
    beam_width: int = 5  # beam search width
    max_mutations: int = 10  # maximum number of mutations
    max_iterations: int = 50  # maximum number of iterations
    min_improvement: float = 0.1  # minimum improvement threshold
    amino_acids: List[str] = None  # allowed amino acids
    forbidden_positions: List[int] = None  # forbidden positions
    batch_size: int = 32  # batch size
    device: str = "cuda:0"
    
    def __post_init__(self):
        if self.amino_acids is None:
            self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                              'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        if self.forbidden_positions is None:
            self.forbidden_positions = []

@dataclass
class Candidate:
    """candidate sequence"""
    sequence: str
    mutations: List[Tuple[int, str, str]]  # (position, original amino acid, mutated amino acid)
    stability_score: float  # cumulative stability score
    ddg_history: List[float]  # ΔΔG value for each step
    
    def __lt__(self, other):
        return self.stability_score > other.stability_score  # maximum heap

class ProteinOptimizer:
    """protein sequence optimizer based on beam search"""
    
    def __init__(self, model_checkpoint: str, config: OptimizationConfig):
        self.config = config
        self.model = self._load_model(model_checkpoint)
        
    def _load_model(self, checkpoint_path: str):
        """load model"""
        print(f"load model: {checkpoint_path}")
        model = LightningDDGModel.load_from_checkpoint(checkpoint_path)
        model.eval().to(self.config.device)
        return model
        
    def _predict_ddg_batch(self, wt_sequences: List[str], mut_sequences: List[str]) -> np.ndarray:
        """batch predict ΔΔG"""
        if len(wt_sequences) != len(mut_sequences):
            raise ValueError("wild type and mutated sequence number mismatch")
            
        # construct batch data
        batch_data = []
        for wt_seq, mut_seq in zip(wt_sequences, mut_sequences):
            batch_data.append({
                'wt_seq': wt_seq,
                'mut_seq': mut_seq,
                'ddG': 0.0  # placeholder
            })
        
       
        batch = tied_featurize_mut(batch_data, device=self.config.device)
        
        with torch.no_grad():
            outputs = self.model(batch)
            ddg_predictions = outputs['ddg'].cpu().numpy()
            
        return ddg_predictions
    
    def _generate_single_mutations(self, sequence: str, forbidden_positions: List[int]) -> List[Tuple[int, str, str]]:
        """generate all possible single point mutations"""
        mutations = []
        for pos in range(len(sequence)):
            if pos in forbidden_positions:
                continue
            original_aa = sequence[pos]
            for new_aa in self.config.amino_acids:
                if new_aa != original_aa:
                    mutations.append((pos, original_aa, new_aa))
        return mutations
    
    def _apply_mutation(self, sequence: str, mutation: Tuple[int, str, str]) -> str:
        """apply mutation to sequence"""
        pos, original_aa, new_aa = mutation
        if sequence[pos] != original_aa:
            print(f"warning: amino acid mismatch at position {pos}: expected {original_aa}, actual {sequence[pos]}")
        
        sequence_list = list(sequence)
        sequence_list[pos] = new_aa
        return ''.join(sequence_list)
    
    def _evaluate_candidates(self, candidates: List[Candidate], wt_sequence: str) -> List[Candidate]:
        """batch evaluate candidates"""
        if not candidates:
            return []
            
        # prepare batch data
        wt_sequences = [wt_sequence] * len(candidates)
        mut_sequences = [candidate.sequence for candidate in candidates]
        
        # batch process to avoid memory overflow
        all_ddg_predictions = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(candidates), batch_size):
            batch_wt = wt_sequences[i:i+batch_size]
            batch_mut = mut_sequences[i:i+batch_size]
            print(f"predict batch {i//batch_size+1}/{len(candidates)//batch_size+1}")
            batch_ddg = self._predict_ddg_batch(batch_wt, batch_mut)
            all_ddg_predictions.extend(batch_ddg)
        
        # update candidate sequence score
        for candidate, ddg in zip(candidates, all_ddg_predictions):
            candidate.ddg_history.append(ddg)
            # stability improvement = -ΔΔG (negative ΔΔG indicates more stable)
            candidate.stability_score += (-ddg)
            
        return candidates
    
    def _select_top_candidates(self, candidates: List[Candidate]) -> List[Candidate]:
        """select top-k candidates"""
        # sort by stability score
        candidates.sort(key=lambda x: x.stability_score, reverse=True)
        return candidates[:self.config.beam_width]
    
    def optimize(self, initial_sequence: str, verbose: bool = True) -> Dict:
        """execute protein sequence optimization"""
        print(f"begin optimization (length: {len(initial_sequence)})")
        print(f"config: beam_width={self.config.beam_width}, max_mutations={self.config.max_mutations}")
        
        # initialize
        initial_candidate = Candidate(
            sequence=initial_sequence,
            mutations=[],
            stability_score=0.0,
            ddg_history=[]
        )
        
        current_beam = [initial_candidate]
        best_candidates_history = []
        iteration = 0
        
        while iteration < self.config.max_iterations and len(current_beam[0].mutations) < self.config.max_mutations:
            if verbose:
                print(f"\n=== iteration {iteration + 1} ===")
                print(f"current beam size: {len(current_beam)}")
            
            # record current best candidates
            best_candidates_history.append(copy.deepcopy(current_beam))
            
            # generate next round candidates
            next_candidates = []
            
            for candidate in current_beam:
                # calculate mutated positions
                mutated_positions = {mut[0] for mut in candidate.mutations}
                forbidden_for_this_candidate = list(mutated_positions) + self.config.forbidden_positions
                
                # generate possible mutations
                possible_mutations = self._generate_single_mutations(
                    candidate.sequence, 
                    forbidden_for_this_candidate
                )
                
                # create new candidate for each possible mutation
                for mutation in possible_mutations:
                    new_sequence = self._apply_mutation(candidate.sequence, mutation)
                    new_candidate = Candidate(
                        sequence=new_sequence,
                        mutations=candidate.mutations + [mutation],
                        stability_score=candidate.stability_score,
                        ddg_history=candidate.ddg_history.copy()
                    )
                    next_candidates.append(new_candidate)
            
            if not next_candidates:
                print("no more possible mutations, stop optimization")
                break
                
            if verbose:
                print(f"generate {len(next_candidates)} candidates")
            
            # batch evaluate candidates
            evaluated_candidates = self._evaluate_candidates(next_candidates, initial_sequence)
            
            # select top-k
            current_beam = self._select_top_candidates(evaluated_candidates)
            
            if verbose:
                print("Top-5 candidates:")
                for i, candidate in enumerate(current_beam[:5]):
                    mut_str = f"{len(candidate.mutations)} mutations" if candidate.mutations else "original sequence"
                    print(f"  {i+1}. stability score: {candidate.stability_score:.3f}, {mut_str}")
                    if candidate.mutations:
                        recent_mutations = candidate.mutations[-3:]  # show recent 3 mutations
                        mut_desc = [f"{m[1]}{m[0]+1}{m[2]}" for m in recent_mutations]
                        print(f"      recent mutations: {', '.join(mut_desc)}")
            
            # check if improvement is enough
            if (iteration > 0 and 
                best_candidates_history[-1][0].stability_score - current_beam[0].stability_score < self.config.min_improvement):
                print(f"improvement less than threshold {self.config.min_improvement}, stop optimization")
                break
                
            iteration += 1
        
        # return result
        best_candidate = current_beam[0]
        
        result = {
            'original_sequence': initial_sequence,
            'optimized_sequence': best_candidate.sequence,
            'mutations': best_candidate.mutations,
            'stability_improvement': best_candidate.stability_score,
            'ddg_history': best_candidate.ddg_history,
            'total_iterations': iteration,
            'optimization_history': best_candidates_history
        }
        
        return result
    
    def analyze_optimization_path(self, result: Dict) -> None:
        """analyze optimization path"""
        print("\n=== optimization result analysis ===")
        print(f"original sequence length: {len(result['original_sequence'])}")
        print(f"optimized sequence length: {len(result['optimized_sequence'])}")
        print(f"total mutations: {len(result['mutations'])}")
        print(f"total stability improvement: {result['stability_improvement']:.3f}")
        print(f"total iterations: {result['total_iterations']}")
        
        print("\nmutation list:")
        for i, (pos, orig, new) in enumerate(result['mutations']):
            ddg = result['ddg_history'][i] if i < len(result['ddg_history']) else 'N/A'
            print(f"  {i+1}. {orig}{pos+1}{new} (ΔΔG: {ddg:.3f})")
        
        print(f"\noptimized sequence:")
        print(result['optimized_sequence'])

def run_optimization_example():
    """run optimization example"""
    # config
    config = OptimizationConfig(
        beam_width=3,
        max_mutations=6,
        max_iterations=10,
        min_improvement=0.05,
        batch_size=1,
        device="cuda:1"
    )
    
    # model checkpoint path
    checkpoint_path = "/home/xy_th/double_mutation_project/unistab/experiment/best-epoch=10-val_pcc=0.8011.ckpt"
    
    # example sequence (you can replace with the sequence you want to optimize)
    initial_sequence = "MSNVKQQTAQIVDWLSSTLGKDHQYREDSLSLTANENYPSALVRLTSGSTAGAFYHCSFPFEVPAGEWHFPEPGHMNAIADQVRDLGKTLIGAQAFDWRPNGGSTAEQALMLAACKPGEGFVHFAHRDGGHFALESLAQKMGIEIFHLPVNPTSLLIDVAKLDEMVRRNPHIRIVILDQSFKLRWQPLAEIRSVLPDSCTLTYDMSHDGGLIMGGVFDSPLSCGADIVHGNTHKTIPGPQKGYIGFKSAQHPLLVDTSLWVCPHLQSNCHAEQLPPMWVAFKEMELFGRDYAAQIVSNAKTLARHLHELGLDVTGESFGFTQTHQVHFAVGDLQKALDLCVNSLHAGGIRSTNIEIPGKPGVHGIRLGVQAMTRRGMKEKDFEVVARFIADLYFKKTEPAKVAQQIKEFLQAFPLAPLAYSFDNYLDEELLAAVYQGAQR"
    
    # create optimizer
    optimizer = ProteinOptimizer(checkpoint_path, config)
    
    import time

    start_time = time.time()
    # execute optimization
    result = optimizer.optimize(initial_sequence, verbose=True)
    
    end_time = time.time()
    print(f"optimization time: {end_time - start_time:.2f} seconds")
    # analyze result
    optimizer.analyze_optimization_path(result)
    
    return result

if __name__ == "__main__":
    result = run_optimization_example()