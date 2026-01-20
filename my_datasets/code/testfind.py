import os, time, shutil

def save_best_val(model, optimizer, scheduler, scaler, step, out_dir):
    best_dir = os.path.join(out_dir, "best_val")
    os.makedirs(best_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"best_step_{step}_{timestamp}.pt"
    path = os.path.join(best_dir, fname)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
    }, path)
    print(f"[🏆] Saved best_val as {path}")

    # update a 'current' alias file (atomic replace)
    current = os.path.join(best_dir, "checkpoint.pt")
    tmp = current + ".tmp"
    shutil.copy(path, tmp)
    os.replace(tmp, current)

    # optional: prune older bests (keep N)
    keep = 5
    bests = sorted([os.path.join(best_dir,f) for f in os.listdir(best_dir) if f.startswith("best_step_")])
    if len(bests) > keep:
        for old in bests[:-keep]:
            try:
                os.remove(old)
            except Exception:
                pass
