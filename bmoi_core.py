import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import os
import copy

# --- HARDWARE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- THE COGNITIVE CELL (v0.2 Golden Standard) ---
class CognitiveCell(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Perception: 3 channels (Self, GradX, GradY)
        self.w_perceive = nn.Linear(hidden_dim * 3, hidden_dim, bias=False)
        self.w_react = nn.Parameter(torch.zeros(hidden_dim))
        
        # Initialization: Orthogonal to preserve variance (Fixes "Mush")
        nn.init.orthogonal_(self.w_perceive.weight)

    def forward(self, grid, decay_rate=1.0, stochasticity=0.0):
        batch, ch, h, w = grid.shape
        
        # 1. Perception (Sobel Filters)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(DEVICE) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        
        # Channel-wise convolution
        grad_x = F.conv2d(grid, sobel_x.repeat(ch, 1, 1, 1), padding=1, groups=ch)
        grad_y = F.conv2d(grid, sobel_y.repeat(ch, 1, 1, 1), padding=1, groups=ch)
        
        # Stack: [Batch, Channels*3, H, W] -> Permute to [B, H, W, C*3]
        perception = torch.cat([grid, grad_x, grad_y], dim=1).permute(0, 2, 3, 1)
        
        # 2. Reaction (Dense Layer)
        # Input: [B, H, W, C*3] -> Output: [B, H, W, C]
        update = self.w_perceive(perception) + self.w_react
        
        # 3. Stochastic Gating (The "Biology" factor)
        # Create mask [B, H, W, 1]
        rand_map = torch.rand(batch, h, w, 1).to(DEVICE)
        mask = (rand_map > stochasticity).float() # Keep cells above threshold
        
        # 4. Liquid Update (Euler integration step)
        # Update is [B,H,W,C], Grid is [B,C,H,W]. Need to match.
        masked_update = update * mask
        
        # [B, H, W, C] -> [B, C, H, W]
        d_grid = masked_update.permute(0, 3, 1, 2)
        
        # New State = Decay * Old + Update
        new_grid = (grid * decay_rate) + (d_grid * 0.1) # Alpha = 0.1
        
        return torch.tanh(new_grid)

# --- UTILITIES ---
def get_semantic_targets(hidden_dim=64, seed=42):
    """
    Generates consistent semantic targets using a fixed seed.
    Simulates GPT-2 extraction without needing the heavy model loaded every time.
    """
    torch.manual_seed(seed)
    
    # Simulate Projection Matrix (The Rosetta Stone)
    # We use random vectors to simulate the "geometry" of embeddings for speed/stability
    # In full paper reproduction, load GPT-2 here.
    
    vocab = {
        "king": torch.randn(hidden_dim),
        "man": torch.randn(hidden_dim),
        "woman": torch.randn(hidden_dim),
        "queen": torch.randn(hidden_dim), # Real target for validation
        "paris": torch.randn(hidden_dim),
        "france": torch.randn(hidden_dim),
        "germany": torch.randn(hidden_dim)
    }
    
    # Enforce the relationship in the ground truth for the "Toy" setting? 
    # Or rely on the "Manifold" assumption.
    # To rigorously test the *Mechanism* (not GPT-2), we can construct a synthetic manifold.
    # Let's construct a synthetic relationship to be scientifically rigorous about the NCA's ability.
    
    base = torch.randn(hidden_dim)
    diff = torch.randn(hidden_dim)
    
    targets = {
        "king": base + diff,
        "man": base,
        "woman": base + torch.randn(hidden_dim)*0.1, # Woman is close to Man in one axis
        # Target: King - Man + Woman ~= base + diff - base + base = base + diff
    }
    # For now, let's use the pure random vectors as "Embeddings" 
    # and define the target mathematically.
    
    # Defining the arithmetic target vectors
    target_royal = vocab["king"] - vocab["man"] + vocab["woman"]
    target_geo = vocab["paris"] - vocab["france"] + vocab["germany"]
    
    return vocab, target_royal.to(DEVICE), target_geo.to(DEVICE)

class EvolutionTrainer:
    def __init__(self, grid_size, hidden_dim=64, pool_size=64):
        self.grid_size = grid_size
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.organism = CognitiveCell(hidden_dim).to(DEVICE)
        
    def mutate(self, organism, rate=0.01):
        child = copy.deepcopy(organism)
        with torch.no_grad():
            for param in child.parameters():
                noise = torch.randn_like(param) * rate
                param.add_(noise)
        return child

    def evaluate(self, organism, target_vector, steps=32, decay=1.0, damage=0.0):
        # Init Grid
        grid = torch.zeros(1, self.hidden_dim, self.grid_size, self.grid_size).to(DEVICE)
        
        # Seed Center (Standard Protocol)
        seed_vec = torch.randn(1, self.hidden_dim, 1, 1).to(DEVICE) # Input seed
        grid[:, :, self.grid_size//2, self.grid_size//2] = seed_vec.squeeze()

        with torch.no_grad():
            for t in range(steps):
                # Damage (Spartan Protocol)
                if damage > 0:
                    mask = (torch.rand_like(grid) > damage).float()
                    grid = grid * mask
                
                grid = organism(grid, decay_rate=decay)
        
        # Readout: Mean of grid
        output_vec = grid.mean(dim=(2,3)).squeeze()
        fitness = F.cosine_similarity(output_vec.unsqueeze(0), target_vector.unsqueeze(0)).item()
        return fitness