import torch
import numpy as np
import random
import os
import json
import time
from collections import defaultdict
from rift_a.envs.rift_world_commitment import RiftWorldCommitment
from rift_a.agents.geometric_commit import GeometricCommitAgent
from rift_a.agents.transformer_predictor import TransformerPredictor, TransformerTrainer
from rift_a.agents.rnn_world_model import RNNWorldModel, RNNTrainer
from rift_a.agents.rl_baseline import RLBaseline, RLTrainer

# 10 Seeds to find survival regime
SEEDS = list(range(100, 110))
TRAIN_EPISODES = 50
EVAL_EPISODES = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPERIMENT_DIR = "rift_a/logs/"
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

def run_experiment_phase_a1():
    print(f"🚀 STARTING PHASE A1 (MINI): 4D Observation Baseline")
    print(f"   Device: {DEVICE}")
    results = defaultdict(dict)

    print("\n--- Testing Geometric Agent ---")
    geo_stats = run_eval_geometric(SEEDS)
    results['Geometric'] = geo_stats

    print("\n--- Training Transformer Agent ---")
    trans_stats = run_train_eval_predictive("Transformer", SEEDS)
    results['Transformer'] = trans_stats

    print("\n--- Training RNN Agent ---")
    rnn_stats = run_train_eval_predictive("RNN", SEEDS)
    results['RNN'] = rnn_stats

    print("\n--- Training RL Agent ---")
    rl_stats = run_train_eval_rl(SEEDS)
    results['RL'] = rl_stats

    print("\n✅ Phase A1 (Mini) Complete.")
    print_summary(results)
    return results

def run_eval_geometric(seeds):
    env = RiftWorldCommitment(device=DEVICE)
    thresholds = [0.1, 0.15, 0.2, 0.25]
    best_survival = -1.0
    best_stats = {}
    for th in thresholds:
        agent = GeometricCommitAgent(trace_threshold=th)
        stats = evaluate_agent(env, agent, seeds, mode="Geometric")
        print(f"   Threshold {th}: Survival Rate = {stats['survival_rate']:.2f}")
        if stats['survival_rate'] >= best_survival:
            best_survival = stats['survival_rate']
            best_stats = stats
            best_stats['best_threshold'] = th
    return best_stats

def run_train_eval_predictive(agent_type, seeds):
    env = RiftWorldCommitment(device=DEVICE)
    final_stats = []
    for seed in seeds:
        if agent_type == "Transformer":
            agent = TransformerPredictor().to(DEVICE)
            trainer = TransformerTrainer(agent, device=DEVICE)
        else:
            agent = RNNWorldModel().to(DEVICE)
            trainer = RNNTrainer(agent, device=DEVICE)
        buffer = []
        for ep in range(TRAIN_EPISODES):
            obs = env.reset(seed=seed + ep*1000)
            episode_data = []
            done = False
            t = 0
            while not done:
                commit, _ = agent.act(obs)
                next_obs, reward, done, info = env.step(commit, t)
                episode_data.append((obs, info['hazard_active']))
                obs = next_obs
                t += 1
            if agent_type == "Transformer": agent.reset_memory()
            else: agent.reset_memory()
            buffer.append(episode_data)
            if len(buffer) >= 10:
                loss = trainer.train_on_batch(buffer)
                buffer = []
        eval_seeds_local = [seed * 10 + i for i in range(10)]
        stats = evaluate_agent(env, agent, eval_seeds_local, mode="Predictive")
        final_stats.append(stats)
    avg_stats = {k: np.mean([s[k] for s in final_stats]) for k in final_stats[0]}
    return avg_stats

def run_train_eval_rl(seeds):
    env = RiftWorldCommitment(device=DEVICE)
    final_stats = []
    for seed in seeds:
        agent = RLBaseline().to(DEVICE)
        trainer = RLTrainer(agent, device=DEVICE)
        for ep in range(TRAIN_EPISODES):
            obs = env.reset(seed=seed + ep*1000)
            done = False
            t = 0
            while not done:
                action, _ = agent.act(obs)
                obs, reward, done, info = env.step(action, t)
                agent.rewards.append(reward)
                t += 1
            trainer.update()
        eval_seeds_local = [seed * 10 + i for i in range(10)]
        stats = evaluate_agent(env, agent, eval_seeds_local, mode="RL")
        final_stats.append(stats)
    avg_stats = {k: np.mean([s[k] for s in final_stats]) for k in final_stats[0]}
    return avg_stats

def evaluate_agent(env, agent, seeds, mode="Geometric"):
    metrics = {'survival_rate': [], 'energy_wasted': [], 'commitment_time': [], 'too_late': [], 'false_negatives': []}
    for seed in seeds:
        obs = env.reset(seed=seed)
        if mode == "Transformer": agent.reset_memory()
        if mode == "RNN": agent.reset_memory()
        done = False
        t = 0
        committed_at = -1
        hazard_start = env.hazard_time
        while not done:
            if mode == "Geometric": commit = agent.act(obs)
            elif mode == "RL":
                probs = agent(obs.unsqueeze(0))
                commit = probs.item() > 0.5
            else: commit, prob = agent.act(obs)
            if commit and committed_at == -1: committed_at = t
            obs, _, done, info = env.step(commit, t)
            t += 1
        survived = (info['health'] > 0)
        metrics['survival_rate'].append(1.0 if survived else 0.0)
        if committed_at != -1 and committed_at < hazard_start:
            metrics['energy_wasted'].append(hazard_start - committed_at)
        else: metrics['energy_wasted'].append(0)
        metrics['commitment_time'].append(committed_at if committed_at != -1 else 100)
        if committed_at == -1 and not survived: metrics['false_negatives'].append(1)
        else: metrics['false_negatives'].append(0)
        if committed_at > hazard_start: metrics['too_late'].append(1)
        else: metrics['too_late'].append(0)
    return {k: np.mean(v) for k, v in metrics.items()}

def print_summary(results):
    print("\n📊 EXPERIMENT SUMMARY (MINI)")
    print(f"{'Agent':<15} | {'Survival':<10} | {'Commit T':<10} | {'Energy Wasted':<15} | {'Fail: Late':<10} | {'Fail: FN':<10}")
    print("-" * 80)
    for agent, stats in results.items():
        print(f"{agent:<15} | {stats['survival_rate']:.2f}       | {stats['commitment_time']:.1f}       | {stats['energy_wasted']:.1f}           | {stats['too_late']:.2f}       | {stats['false_negatives']:.2f}")

if __name__ == "__main__":
    run_experiment_phase_a1()
