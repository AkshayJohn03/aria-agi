from tensorboard.backend.event_processing import event_accumulator
import os

log_dir = r"D:\aria\aria_ai\aria_ai_assistant\artifacts\runs"  # or where your runs are
best_val = float("inf")
best_step = None
best_file = None

for root, _, files in os.walk(log_dir):
    for f in files:
        if "tfevents" in f:
            path = os.path.join(root, f)
            ea = event_accumulator.EventAccumulator(path)
            ea.Reload()
            for tag in ("eval/cross_entropy", "eval/loss"):
                if tag in ea.Tags().get("scalars", []):
                    for s in ea.Scalars(tag):
                        if s.value < best_val:
                            best_val = s.value
                            best_step = s.step
                            best_file = path

print(f"✅ Best val: {best_val:.4f} at step {best_step}")
print(f"📂 File: {best_file}")
