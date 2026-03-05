import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

log_files = [
    ("/mnt/data/flashinernimage_t.log.json", "FlashInternImage-T"),
    ("/mnt/data/flashinternimage_b.log.json", "FlashInternImage-B"),
    ("/mnt/data/resnet50.log.json", "ResNet-50"),
    ("/mnt/data/resnet101.log.json", "ResNet-101"),
]

# --- style (Times-like) ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "axes.linewidth": 1.2,
})

def load_jsonlog(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rec = {}
            for k, v in obj.items():
                if isinstance(v, (int, float, str)) or v is None:
                    rec[k] = v
            rows.append(rec)
    return pd.DataFrame(rows)

def compute_step(train_df):
    g = train_df.groupby("epoch")["iter"].max().dropna()
    if len(g) == 0:
        return None, None
    max_iter = int(g.mode().iloc[0])
    train_df = train_df.copy()
    train_df["step"] = (train_df["epoch"].astype(int) - 1) * max_iter + train_df["iter"].astype(int)
    return train_df, max_iter

def pick_xlim_near_best(x_best, x_max, start=0, pad_frac=0.15, min_pad=200):
    if x_best is None or (isinstance(x_best, float) and (not np.isfinite(x_best))):
        return (start, x_max)
    pad = max(min_pad, int(round(x_best * pad_frac)))
    end = min(x_max, int(round(x_best + pad)))
    end = max(end, start + 10)
    return (start, end)

def rolling_smooth(s, window=9):
    return s.rolling(window, center=True, min_periods=1).mean()

letters = list("abcdefghijklmnopqrstuvwxyz")

# First pass: prepare data for each model + compute global mAP y-limits
prepared = []
all_map_vals = []

for path, name in log_files:
    df = load_jsonlog(path)

    # train
    train = df[df.get("mode").astype(str).str.lower().eq("train")].copy()
    train = train.dropna(subset=["epoch", "iter"]).sort_values(["epoch", "iter"])
    train, _ = compute_step(train)
    if train is None:
        raise RuntimeError(f"Could not compute steps for {path}")

    loss_keys = ["loss", "loss_cls_classes", "loss_bbox", "loss_mask"]
    loss_keys = [k for k in loss_keys if k in train.columns]
    for k in loss_keys:
        train[k] = pd.to_numeric(train[k], errors="coerce")
        train[k] = rolling_smooth(train[k], window=9)

    min_loss_step = float(train.loc[train["loss"].idxmin(), "step"]) if "loss" in train.columns and train["loss"].notna().any() else None
    max_step = int(train["step"].max())
    loss_xlim = pick_xlim_near_best(min_loss_step, max_step, start=0, pad_frac=0.20, min_pad=500)

    # val
    val = df[df.get("mode").astype(str).str.lower().isin(["val", "eval", "test"])].copy()
    bbox_map = "bbox_mAP" if "bbox_mAP" in val.columns else ("coco/bbox_mAP" if "coco/bbox_mAP" in val.columns else None)
    segm_map = "segm_mAP" if "segm_mAP" in val.columns else ("coco/segm_mAP" if "coco/segm_mAP" in val.columns else None)
    map_fields = [c for c in [bbox_map, segm_map] if c is not None]

    val_ep = pd.DataFrame()
    best_epoch = None
    map_xlim = None
    if len(val) and "epoch" in val.columns and map_fields:
        val["epoch"] = pd.to_numeric(val["epoch"], errors="coerce")
        val = val.dropna(subset=["epoch"])
        cols = ["epoch"] + map_fields
        val2 = val[cols].copy()
        for c in map_fields:
            val2[c] = pd.to_numeric(val2[c], errors="coerce")
        val_ep = val2.groupby("epoch", as_index=False).last().sort_values("epoch")
        # best epoch on bbox if available else segm
        key = bbox_map if bbox_map in val_ep.columns else (segm_map if segm_map in val_ep.columns else None)
        if key is not None and val_ep[key].notna().any():
            best_epoch = int(val_ep.loc[val_ep[key].idxmax(), "epoch"])
        max_epoch = int(val_ep["epoch"].max())
        map_xlim = pick_xlim_near_best(best_epoch, max_epoch, start=1, pad_frac=0.25, min_pad=5)

        # collect y values within displayed x-range for global ylim
        x0, x1 = map_xlim
        sub = val_ep[(val_ep["epoch"] >= x0) & (val_ep["epoch"] <= x1)]
        for c in map_fields:
            all_map_vals.extend(sub[c].dropna().tolist())

    prepared.append({
        "name": name,
        "train": train,
        "loss_keys": loss_keys,
        "min_loss_step": min_loss_step,
        "loss_xlim": loss_xlim,
        "val_ep": val_ep,
        "bbox_map": bbox_map,
        "segm_map": segm_map,
        "map_fields": map_fields,
        "best_epoch": best_epoch,
        "map_xlim": map_xlim,
    })

# global y-limits for mAP axes
if all_map_vals:
    y_min = float(min(all_map_vals))
    y_max = float(max(all_map_vals))
    pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    map_ylim = (y_min - pad, y_max + pad)
else:
    map_ylim = None

# -------- Plot: 4 rows x 2 cols --------
fig, axs = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)

for i, item in enumerate(prepared):
    name = item["name"]

    # (left) training losses
    ax = axs[i, 0]
    tr = item["train"]
    loss_keys = item["loss_keys"]
    x0, x1 = item["loss_xlim"]
    for k in loss_keys:
        ax.plot(tr["step"], tr[k], linewidth=2.5, label=k)
    if item["min_loss_step"] is not None:
        ax.axvline(item["min_loss_step"], linestyle="--", linewidth=2.0)
    ax.set_xlim(x0, x1)

    # log scale if needed in shown region
    if "loss" in tr.columns:
        y = tr.loc[(tr["step"] >= x0) & (tr["step"] <= x1), "loss"].dropna()
        if len(y) and (y.max() / max(y.min(), 1e-12) > 20):
            ax.set_yscale("log")

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.grid(False)
    ax.set_title(f"{name} — Training losses", pad=8)

    # legend above axis to avoid vline intersection
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0.0)

    # subplot label only
    ax.text(0.01, 0.98, f"({letters[2*i]})", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

    # (right) validation mAP
    ax = axs[i, 1]
    val_ep = item["val_ep"]
    if len(val_ep) and item["map_fields"]:
        x0, x1 = item["map_xlim"]
        # bbox green, segm red
        if item["bbox_map"] and item["bbox_map"] in val_ep.columns:
            ax.plot(val_ep["epoch"], val_ep[item["bbox_map"]], linewidth=2.5, marker="o", markersize=4,
                    color="green", label="bbox mAP")
        if item["segm_map"] and item["segm_map"] in val_ep.columns:
            ax.plot(val_ep["epoch"], val_ep[item["segm_map"]], linewidth=2.5, marker="o", markersize=4,
                    color="red", label="segm mAP")

        if item["best_epoch"] is not None:
            ax.axvline(item["best_epoch"], linestyle="--", linewidth=2.0)

        ax.set_xlim(x0, x1)
        if map_ylim is not None:
            ax.set_ylim(*map_ylim)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.grid(False)
        ax.set_title(f"{name} — Validation COCO-style metrics", pad=8)

        # legend above axis (avoid dashed line)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0.0)
    else:
        ax.text(0.5, 0.5, "No validation mAP found in log", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    ax.text(0.01, 0.98, f"({letters[2*i+1]})", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

out_png = "/mnt/data/all_models_loss_map_auto_v2.png"
out_pdf = "/mnt/data/all_models_loss_map_auto_v2.pdf"
plt.savefig(out_png, dpi=300)
plt.savefig(out_pdf)
plt.show()

(out_png, out_pdf, map_ylim)
