"""Training loop for every condition: early stopping on validation CE, a checkpoint after each epoch, and resume.

Each run lives in its own folder (e.g. Drive runs/r4_s0/): last.pt is rewritten every epoch and is what a
restarted run resumes from; best.pt holds the weights with the lowest validation CE (used for the one test eval).
Randomness is reseeded from (seed, epoch) at the start of every epoch, so a resumed run is identical to an
uninterrupted one without having to save RNG state.

Self-check: python -m src.train  (tiny CPU run: trains 3 epochs straight, then kills at epoch 1 and resumes)
"""
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def epoch_order(n, seed, epoch):
    """Shuffle order for one epoch. Shared with Paddy's frozen trainer so seed s sees the same order (decision 3)."""
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + epoch)).tolist()


def trainable_state(model):
    """Only LoRA + head weights: a few MB, not the 346 MB frozen ViT."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save(obj, path):
    # write then rename, so a disconnect mid-save never leaves a half-written checkpoint
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean cross-entropy (nats) over the loader, with dropout off."""
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        logits = model(pixel_values=x.to(device)).logits
        total += F.cross_entropy(logits, y.to(device), reduction="sum").item()
        n += len(y)
    return total / n


def train(model, train_ds, val_ds, out_dir, seed, lr, batch_size=32, weight_decay=0.01,
          max_epochs=50, patience=5, num_workers=2, device=None, stop_after=None):
    """Train until validation CE hasn't improved for `patience` epochs. Resumes from out_dir/last.pt if present.

    stop_after: return after this epoch as if Colab disconnected (only used by the self-check).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, num_workers=num_workers)

    state = {"epoch": 0, "best_val_ce": float("inf"), "best_epoch": 0, "bad_epochs": 0,
             "history": [], "train_time_s": 0.0, "done": False}
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        assert not unexpected, f"checkpoint doesn't match this model: {unexpected[:3]}"
        opt.load_state_dict(ckpt["optimizer"])
        state = ckpt["state"]
        print(f"resumed {out.name} after epoch {state['epoch']}")

    while not state["done"]:
        epoch = state["epoch"] + 1
        t0 = time.time()
        torch.manual_seed(seed * 1000 + epoch)  # fixes this epoch's orientations and dropout
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=epoch_order(len(train_ds), seed, epoch),
                            num_workers=num_workers)
        model.train()
        train_loss = 0.0
        for x, y in loader:
            loss = F.cross_entropy(model(pixel_values=x.to(device)).logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(y)

        val_ce = evaluate(model, val_loader, device)
        if val_ce < state["best_val_ce"]:
            state.update(best_val_ce=val_ce, best_epoch=epoch, bad_epochs=0)
            save(trainable_state(model), out / "best.pt")
        else:
            state["bad_epochs"] += 1
        state["epoch"] = epoch
        state["train_time_s"] += time.time() - t0
        state["history"].append({"epoch": epoch, "train_ce": train_loss / len(train_ds), "val_ce": val_ce})
        state["done"] = state["bad_epochs"] >= patience or epoch >= max_epochs
        save({"model": trainable_state(model), "optimizer": opt.state_dict(), "state": state}, out / "last.pt")
        print(f"epoch {epoch}: train CE {train_loss / len(train_ds):.4f}, val CE {val_ce:.4f}"
              f"{'  (best)' if state['best_epoch'] == epoch else ''}")

        if stop_after and epoch >= stop_after:
            return state
    return state


if __name__ == "__main__":
    import tempfile

    from src.data import EuroSATDataset, read_split
    from src.model import build_model
    from src.transforms import eval_transform, train_transform

    csv = "data/splits/split_temp.csv"
    tiny = lambda split: read_split(csv, split).groupby("label").sample(2, random_state=0)  # ponytail: 20 images, CPU-sized
    train_ds, val_ds = EuroSATDataset(tiny("train"), train_transform), EuroSATDataset(tiny("val"), eval_transform)
    kw = dict(seed=0, lr=1e-3, max_epochs=3, patience=10, num_workers=0, device="cpu")

    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        model_a = build_model(4, seed=0)
        full = train(model_a, train_ds, val_ds, a, **kw)

        killed = train(build_model(4, seed=0), train_ds, val_ds, b, stop_after=1, **kw)
        assert killed["epoch"] == 1 and not killed["done"]
        model_b = build_model(4, seed=0)  # fresh process after a disconnect: new model, same folder
        resumed = train(model_b, train_ds, val_ds, b, **kw)

        assert [h["val_ce"] for h in resumed["history"]] == [h["val_ce"] for h in full["history"]], "histories differ"
        sa, sb = trainable_state(model_a), trainable_state(model_b)
        assert all(torch.equal(sa[k], sb[k]) for k in sa), "final weights differ after resume"
        assert (Path(b) / "best.pt").exists() and resumed["best_epoch"] == full["best_epoch"]

    print("train OK: killed-and-resumed run matches the uninterrupted run exactly")
