# find_and_restore_best.py
import os, shutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_DIR = "artifacts/zia_dense_runs/runs"     # change if different
CKPT_DIR = "artifacts/zia_dense_runs/checkpoints"
OUT_PATH = "artifacts/zia_dense_runs/best_val_recovered_checkpoint.pt"

def find_best_from_events(runs_dir):
    best_val = float("inf")
    best_step = None
    for root,dirs,files in os.walk(runs_dir):
        for f in files:
            if not f.startswith("events.out.tfevents"):
                continue
            path = os.path.join(root,f)
            try:
                ea = EventAccumulator(path, size_guidance={"scalars": 0})
                ea.Reload()
                tags = ea.Tags().get("scalars", [])
                # try common tags
                for tag in ("eval/cross_entropy","eval/loss","eval/cross_entropy_epoch","eval/loss_epoch"):
                    if tag in tags:
                        for scalar in ea.Scalars(tag):
                            if scalar.value < best_val:
                                best_val = scalar.value
                                best_step = int(scalar.step)
            except Exception as e:
                print("failed to read", path, e)
    return best_step, best_val

if __name__ == "__main__":
    best_step, best_val = find_best_from_events(RUNS_DIR)
    print("Best from TB events:", best_step, best_val)
    if best_step is None:
        print("No matching eval scalar found. Look manually at runs/ directory.")
        exit(1)
    # look for checkpoint file
    found = None
    for fname in os.listdir(CKPT_DIR):
        if f"step_{best_step}" in fname:
            found = os.path.join(CKPT_DIR, fname)
            break
    if not found:
        print("Checkpoint file for step not found in checkpoints/. Listing nearby files:")
        print(sorted(os.listdir(CKPT_DIR))[-20:])
        exit(1)
    print("Found checkpoint:", found)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    shutil.copy(found, OUT_PATH)
    print(f"Copied -> {OUT_PATH}. You can now use this as your base checkpoint.")
