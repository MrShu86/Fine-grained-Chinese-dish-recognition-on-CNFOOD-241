# -*- coding: utf-8 -*-
"""
Summarize repeated training runs saved by run_repeated_experiments.py.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_best(run_dir):
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        return None
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "best_top1_percent": ckpt.get("best_top1") if isinstance(ckpt, dict) else None,
    }


def seed_from_dir(path):
    m = re.search(r"seed_(\d+)", path.name)
    return int(m.group(1)) if m else None


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
        "min": float(arr.min()) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="repeated_runs")
    parser.add_argument("--out", default="repeated_runs_summary.csv")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for model_dir in sorted(root.glob("*")):
        if not model_dir.is_dir():
            continue
        for run_dir in sorted(model_dir.glob("seed_*")):
            row = read_best(run_dir)
            if row is None:
                rows.append({
                    "model": model_dir.name,
                    "seed": seed_from_dir(run_dir),
                    "run_dir": str(run_dir),
                    "checkpoint": "",
                    "epoch": None,
                    "best_top1_percent": None,
                    "status": "missing_best_pt",
                })
                continue
            row["model"] = model_dir.name
            row["seed"] = seed_from_dir(run_dir)
            row["status"] = "ok"
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        ordered = ["model", "seed", "status", "best_top1_percent", "epoch", "checkpoint", "run_dir"]
        df = df[[c for c in ordered if c in df.columns]]
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    summaries = {}
    if not df.empty:
        for model, group in df[df["status"] == "ok"].groupby("model"):
            summaries[model] = summarize(group["best_top1_percent"].dropna().to_numpy())
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print("[SAVE]", Path(args.out).resolve())


if __name__ == "__main__":
    main()
