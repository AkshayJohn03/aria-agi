import torch
import pandas as pd
import numpy as np
import copy
from bmoi_core import CognitiveCell, EvolutionTrainer, get_semantic_targets, DEVICE

# --- CONFIG ---
SEEDS = [42, 101, 202, 303, 555] # 5 Independent Runs
GENS = 500
GRID_SIZE = 16

results = []

print("🧪 STARTING RIGOROUS BENCHMARK SUITE (v0.2.1 - Fix)...")

for seed in SEEDS:
    print(f"\n🌱 RUNNING SEED {seed}...")
    
    # 1. Get Consistent Targets (The Language)
    # This sets global seed to 42 temporarily to build the dictionary
    vocab, target_royal, _ = get_semantic_targets(seed=42) 
    
    # 2. CRITICAL FIX: Re-Seed for the Organism (The Diversity)
    # We force the global seed to change NOW, so the organism is unique
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 3. Init Trainer (Now using the unique seed)
    trainer = EvolutionTrainer(GRID_SIZE)
    organism = trainer.organism
    
    # 4. Training Loop
    for gen in range(GENS):
        # Mutate
        child = trainer.mutate(organism, rate=0.02)
        
        # Evaluate Child
        fit_child = trainer.evaluate(child, target_royal)
        
        # Evaluate Parent
        fit_parent = trainer.evaluate(organism, target_royal)
        
        if fit_child > fit_parent:
            organism = child
            current_fit = fit_child
        else:
            current_fit = fit_parent
            
        if gen % 100 == 0:
            results.append({
                "seed": seed,
                "generation": gen,
                "fitness": current_fit,
                "type": "Genesis_Baseline"
            })

print("\n📊 BENCHMARK COMPLETE.")
df = pd.DataFrame(results)
df.to_csv("bmoi_benchmark_results.csv", index=False)

# Calculate Stats
final_stats = df[df["generation"] == (GENS - (GENS % 100)) if GENS % 100 != 0 else GENS-100]
# Use the last logged generation (e.g., 400 for 500 steps if logging every 100)
# Actually, let's just grab the max generation for each seed
max_gen = df["generation"].max()
final_stats = df[df["generation"] == max_gen].groupby("type")["fitness"].agg(["mean", "std", "min", "max"])

print("\n📋 FINAL STATISTICS (Corrected):")
print(final_stats)