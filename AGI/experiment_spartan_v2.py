import torch
import pandas as pd
import numpy as np
import copy
from bmoi_core import CognitiveCell, EvolutionTrainer, get_semantic_targets, DEVICE

# --- CONFIG ---
SEED = 42
GENS = 500
GRID_SIZE = 16
DAMAGE_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9] # The Stress Test

print("🛡️ LAUNCHING SPARTAN PROTOCOL V2 (Comparative Stress Test)...")

# 1. Setup
torch.manual_seed(SEED)
np.random.seed(SEED)
vocab, target_royal, _ = get_semantic_targets(seed=42)

# --- TRAIN NAIVE (Control Group) ---
print("\n👶 Training Naive Organism (No Damage)...")
trainer_naive = EvolutionTrainer(GRID_SIZE)
naive_org = trainer_naive.organism
best_naive_fit = -1.0

for gen in range(GENS):
    child = trainer_naive.mutate(naive_org, rate=0.02)
    fit_child = trainer_naive.evaluate(child, target_royal, damage=0.0) # Clean env
    fit_parent = trainer_naive.evaluate(naive_org, target_royal, damage=0.0)
    
    if fit_child > fit_parent:
        naive_org = child
        best_naive_fit = fit_child

print(f"   Naive Training Complete. Best Fitness: {best_naive_fit:.3f}")

# --- TRAIN SPARTAN (Experimental Group) ---
print("\n⚔️ Training Spartan Organism (20% Chronic Damage)...")
trainer_spartan = EvolutionTrainer(GRID_SIZE)
spartan_org = trainer_spartan.organism
best_spartan_fit = -1.0

for gen in range(GENS):
    child = trainer_spartan.mutate(spartan_org, rate=0.02)
    # Train in harsh environment
    fit_child = trainer_spartan.evaluate(child, target_royal, damage=0.2) 
    fit_parent = trainer_spartan.evaluate(spartan_org, target_royal, damage=0.2)
    
    if fit_child > fit_parent:
        spartan_org = child
        best_spartan_fit = fit_child

print(f"   Spartan Training Complete. Best Fitness: {best_spartan_fit:.3f}")

# --- THE STRESS TEST ---
print("\n📉 RUNNING DAMAGE CURVES...")
results = []

for dmg in DAMAGE_LEVELS:
    # Test Naive
    # Run 10 trials to get average performance at this damage level
    naive_scores = [trainer_naive.evaluate(naive_org, target_royal, damage=dmg) for _ in range(10)]
    naive_mean = np.mean(naive_scores)
    
    # Test Spartan
    spartan_scores = [trainer_spartan.evaluate(spartan_org, target_royal, damage=dmg) for _ in range(10)]
    spartan_mean = np.mean(spartan_scores)
    
    print(f"   Damage {int(dmg*100)}%: Naive={naive_mean:.3f} | Spartan={spartan_mean:.3f}")
    
    results.append({"Damage": dmg, "Type": "Naive", "Fitness": naive_mean})
    results.append({"Damage": dmg, "Type": "Spartan", "Fitness": spartan_mean})

# Save Data
df = pd.DataFrame(results)
df.to_csv("bmoi_spartan_results.csv", index=False)
print("\n✅ Data saved to bmoi_spartan_results.csv")