"""Regenerate the report figures as vector PDF into latex/figures/.

Reuses src/ and the notebook plotting code against the on-disk cache and
checkpoints (already present). Does NOT re-extract zips or re-cache images.
Run: .venv/bin/python latex/make_figures.py
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# src/data.py uses cwd-relative "../data/..." paths -> run as if from notebooks/.
os.chdir(ROOT / "notebooks")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import make_loaders, pad_to_square, apply_clahe
from src.model import load_checkpoint
from src.metrics import mae

OUT = ROOT / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / name, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_preprocessing():
    raw = cv2.imread(str(ROOT / "data/raw/train/13624.png"), cv2.IMREAD_GRAYSCALE)
    padded = pad_to_square(raw)
    resized = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_AREA)
    clahe = apply_clahe(resized)
    imgs = [raw, padded, resized, clahe]
    titles = [f"original {raw.shape[1]}x{raw.shape[0]}",
              f"padded {padded.shape[1]}x{padded.shape[0]}",
              f"resized {resized.shape[1]}x{resized.shape[0]}", "CLAHE"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap="gray")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "preprocessing.pdf")


def fig_loss_history(ck):
    th, vh = ck["train_loss_history"], ck["val_loss_history"]
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    ax.plot(th, label="train", color="#2b6cb0", lw=2)
    ax.plot(vh, label="valid", color="#e07b39", lw=2)
    best_ep = int(np.argmin(vh))
    ax.scatter(best_ep, vh[best_ep], color="#e07b39", zorder=5, s=40,
               ec="white", label=f"best valid (ep {best_ep})")
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("loss (L1, standardized)", fontsize=11)
    ax.set_title("Loss history", fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=True, fontsize=10)
    ax.margins(x=0.01)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "loss_history.pdf")


def fig_mae_by_age(model, loaders, scaler):
    import pandas as pd
    y_true, y_pred, _ = model.predict_group(loaders["test"], scaler)
    tbl = pd.DataFrame({"true": y_true, "abs_err": np.abs(y_pred - y_true)})
    tbl["age_bin"] = pd.cut(tbl["true"], bins=[0, 36, 72, 108, 144, 180, 240],
                            labels=["0-3y", "3-6y", "6-9y", "9-12y", "12-15y", "15y+"])
    by_bin = tbl.groupby("age_bin", observed=True)["abs_err"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    by_bin.plot(kind="bar", ax=ax, color="tab:orange")
    ax.set_ylabel("mean abs error (months)")
    ax.set_xlabel("")
    ax.set_title("Test MAE by age group")
    fig.tight_layout()
    save(fig, "mae_by_age.pdf")


def main():
    fig_preprocessing()
    loaders, scaler = make_loaders()
    model, ck = load_checkpoint(ROOT / "data/checkpoints/adam.pt")
    fig_loss_history(ck)
    fig_mae_by_age(model, loaders, scaler)


if __name__ == "__main__":
    main()
