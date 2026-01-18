import torch
import numpy as np
import random
from rift_a.envs.rift_world_commitment import RiftWorldCommitment

DEVICE = "cpu"

def diagnose_trace():
    print("🔬 DIAGNOSTIC: Trace Signal Dynamics (Frozen Config)")

    # Initialize Env
    env = RiftWorldCommitment(device=DEVICE)
    obs = env.reset(seed=42)

    cue_time = env.cue_time
    hazard_time = env.hazard_time

    print(f"   Config: Cue @ {cue_time}, Hazard @ {hazard_time}")
    print(f"   {'T':<4} | {'Trace (Mean)':<12} | {'Cue':<5} | {'Hazard':<6}")
    print("-" * 45)

    peak_trace = 0.0
    steps_above_01 = 0

    for t in range(100):
        # We don't commit, just observe physics
        obs, _, _, info = env.step(action_commit=False, step_idx=t)

        # obs[1] is the trace channel (noisy)
        trace_val = obs[1].item()

        is_cue = (t == cue_time)
        is_hazard = (t >= hazard_time and t < hazard_time + 5)

        if trace_val > peak_trace: peak_trace = trace_val
        if trace_val > 0.1: steps_above_01 += 1

        # Print window around events
        if (t >= cue_time - 2 and t <= cue_time + 10) or (t >= hazard_time - 2 and t <= hazard_time + 2):
            print(f"   {t:<4} | {trace_val:.4f}       | {is_cue!s:<5} | {is_hazard!s:<6}")

    print("-" * 45)
    print(f"   MAX Trace Value: {peak_trace:.4f}")
    print(f"   Steps > 0.1: {steps_above_01}")

if __name__ == "__main__":
    diagnose_trace()
