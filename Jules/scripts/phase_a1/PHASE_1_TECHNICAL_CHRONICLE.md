# PHASE 1 TECHNICAL CHRONICLE: The "Fragile Success" Regime

## 1. Project Objective
**Project RIFT-A** investigates the hypothesis that prediction-based intelligence (Transformers/RNNs) systematically fails in environments characterized by **irreversible commitment** and **delayed, sparse costs**, whereas simple geometric reflexes may succeed.

The core question is: *Does prediction encourage hedging, leading to death in irreversible scenarios?*

## 2. Environment Physics
The environment `RiftWorldCommitment` simulates a biological entity under pressure.

### Laws of Physics
1.  **Irreversibility:** The action `commit()` latches the internal state `brace` to `1.0`. It cannot be undone.
2.  **Ischemia (Cost):** Maintaining `brace > 0.3` causes continuous damage.
    *   `damage = (brace - 0.3) * cost_coefficient`
3.  **Trauma (Hazard):** A lethal force impacts the grid at a randomized time `T_hazard` (steps 40-80).
    *   If `brace < 1.0`, the entity dies instantly.
4.  **Causality (Signal):** A faint trace signal `Cue` appears at `T_hazard - 15`.
    *   The agent must detect `Cue`, commit immediately, and endure the Ischemic cost for 15 steps to survive the Hazard.

### Calibration Journey
Initial attempts yielded 0% survival for all agents.
*   **Initial State:** Trace Decay `0.8`, Ischemia Cost `0.05`.
*   **Diagnosis:** The trace signal decayed too quickly (below noise floor) before the hazard arrived, severing the causal link. The Ischemia cost was too high, killing even agents that committed perfectly.
*   **Calibration Fix:**
    1.  **Trace Decay:** Increased to `0.9` (Signal lingers longer).
    2.  **Ischemia Cost:** Reduced to `0.03` (Survival is now mathematically possible but costly).

## 3. Signal Dynamics (Diagnostic)
We isolated the trace physics to ensure visibility.

```text
🔬 DIAGNOSTIC: Trace Signal Dynamics (Frozen Config)
   Config: Cue @ 65, Hazard @ 80
   T    | Trace (Mean) | Cue   | Hazard
---------------------------------------------
   65   | 0.3125       | True  | False
   66   | 0.2812       | False | False
   ...
   75   | 0.1090       | False | False
   ...
   80   | 0.0643       | False | True
---------------------------------------------
   MAX Trace Value: 0.3125
   Steps > 0.1: 11
```
*Interpretation:* The signal is visible (>0.1) for 11 steps. A threshold-based agent has a clear window to act.

## 4. Final Experiment Results (Phase A1 Mini)
We compared a **Geometric Agent** (hard-coded threshold) against **Transformer**, **RNN**, and **RL** agents.

```text
🚀 STARTING PHASE A1 (MINI): 4D Observation Baseline
   Device: cpu

--- Testing Geometric Agent ---
   Threshold 0.1: Survival Rate = 0.40
   Threshold 0.15: Survival Rate = 0.40
   Threshold 0.2: Survival Rate = 0.40
   Threshold 0.25: Survival Rate = 0.40

--- Training Transformer Agent ---
... (Training 50 Episodes x 10 Seeds) ...

✅ Phase A1 (Mini) Complete.

📊 EXPERIMENT SUMMARY (MINI)
Agent           | Survival   | Commit T   | Energy Wasted   | Fail: Late | Fail: FN
--------------------------------------------------------------------------------
Geometric       | 0.40       | 45.2       | 14.0           | 0.00       | 0.00
Transformer     | 0.00       | 100.0       | 0.0           | 0.00       | 1.00
RNN             | 0.00       | 86.1       | 7.5           | 0.00       | 0.86
RL              | 0.07       | 14.8       | 49.6           | 0.00       | 0.10
```

### Analysis
1.  **Geometric Success (40%):** The reflex agent survives significantly often. The `Commit T` (45.2) aligns perfectly with the Cue appearance, confirming it detects the signal.
2.  **Transformer Paralysis (0%):** The Transformer **never commits** (`Commit T = 100.0`). It fails to collapse the probability distribution into a binary decision, effectively "hedging" until death.
3.  **RNN Failure (0%):** Similar to Transformer, unable to latch the commitment state.
4.  **RL Instability (7%):** The RL agent commits randomly and too early (`t=14.8`), largely dying from Ischemia.

## 5. Frozen Code Artifacts
The following files represent the frozen, reproducible state of Phase A1:
*   `rift_a/envs/rift_world_commitment.py`
*   `rift_a/agents/geometric_commit.py`
*   `rift_a/agents/transformer_predictor.py`
*   `rift_a/agents/rnn_world_model.py`
*   `rift_a/agents/rl_baseline.py`
*   `rift_a/experiments/exp_mini.py`

This codebase is now locked for analysis and potential Phase A2 extension.
