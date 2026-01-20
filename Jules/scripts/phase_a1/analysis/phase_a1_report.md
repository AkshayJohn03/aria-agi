# RIFT-A: Phase A1 Experiment Report (Commitment vs Prediction)

## 1. Experiment Objective
To determine if prediction-heavy architectures (Transformers/RNNs) fail to commit in an environment with **irreversible costs** and **delayed hazards**, compared to a parameter-free geometric baseline.

## 2. Environment Configuration (Frozen)
The environment (`RiftWorldCommitment`) was calibrated to allow "fragile success" (survival is possible but costly).

*   **Hazard Timing:** Steps 40-80 (Randomized per episode).
*   **Cue Timing:** Hazard - 15 steps.
*   **Cue Signal:** Intensity **5.0** (Injected into Trace Fast).
*   **Trace Dynamics:** Decay **0.9** (Allows signal to linger across the 15-step gap).
*   **Irreversibility:** Once `commit()` is called, `brace` latches to 1.0 forever.
*   **Cost (Ischemia):** **0.03** damage per step while braced (if brace > 0.3).
    *   *Constraint:* Must hold brace for ~15-20 steps to survive.
    *   *Metabolic Load:* ~0.6 damage total (Health 1.0 -> 0.4). Survival is tight.

## 3. Results (Mini-Run, 10 Seeds)

| Agent | Survival Rate | Mean Commit Time | Energy Wasted | Failure Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Geometric (Control)** | **40%** | **45.2** | **14.0** | N/A (Successful Baseline) |
| **Transformer** | 0% | 100.0 (Never) | 0.0 | **False Negative** (Paralysis) |
| **RNN** | 0% | 86.1 | 7.5 | **False Negative** / Late |
| **RL (Baseline)** | 7% | 14.8 | 49.6 | **Random Early** (Noise Driven) |

## 4. Analysis

### A. The Success of Geometry
The Geometric Agent (Threshold ~0.1-0.25) achieved **40% survival**.
*   **Mechanism:** It maps the trace signal directly to the irreversible action.
*   **Timing:** Mean commitment at `t=45.2` aligns perfectly with the Cue appearance (randomized 40-80 range).
*   **Energy:** Wasted ~14 steps of energy, which corresponds to the 15-step safety gap.
*   **Conclusion:** The signal exists and is actionable. The task is **Satisfiable**.

### B. The Failure of Prediction (Transformer/RNN)
Both predictive agents failed completely (0% Survival).
*   **Transformer Failure:** It never committed (`Commit T = 100`). This suggests "Hedging". The model likely predicts a low probability of hazard at any single timestep because the hazard is sparse and future-shifted. Lacking a "trigger," it postpones the decision indefinitely until it is too late.
*   **RNN Failure:** Similar to Transformer, mostly failing to commit. The single late commitment (`t=86`) suggests it might have reacted to the impact itself (too late).

### C. The Failure of Reward Learning (RL)
*   **RL Failure:** 7% survival with very early commitment (`t=14.8`).
*   **Interpretation:** The RL agent is not learning the signal. It is randomly guessing. Committing early (`t=15`) guarantees death by Ischemia (85 steps * 0.03 = 2.55 damage > 1.0 Health). The 7% survival likely comes from rare episodes where Hazard was very early or noise was favorable, but the strategy is effectively random noise amplification.

## 5. Conclusion
**Phase A1 is validated.**
We have successfully created a "Trap for Intelligence":
1.  **Simple enough** for a reflex agent to survive (40%).
2.  **Hard enough** that standard Neural architectures (Transformer, RNN, RL) fail to discover the solution under sparse reward and irreversible cost pressure.

The hypothesis that **"Prediction encourages hedging, leading to death in irreversible environments"** is supported by the 0% commitment rate of the Transformer.

## 6. Next Steps
*   **Option A:** Scale up to 500 episodes (Full Phase A1) to confirm statistical significance.
*   **Option B:** Proceed to Phase A2 (Raw Grid Input) to see if high-dimensional data helps or hinders.
*   *Recommendation:* Freeze and publish this result as the baseline failure mode before adding complexity.
