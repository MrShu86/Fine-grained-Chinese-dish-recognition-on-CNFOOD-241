# -*- coding: utf-8 -*-
"""Compute paired significance statistics for repeated-run evaluations.

Inputs are the 10 evaluation folders under:
  repaet run eval/best_baseseed{seed}
  repaet run eval/best_fullseed{seed}

This script is used for the revised manuscript's five-seed statistical
validation. It pools paired seed-image observations after checking that the
baseline and full-model prediction files have the same image paths and labels.

Outputs:
  repaet run eval/significance_stats/significance_summary.csv
  repaet run eval/significance_stats/significance_summary.md
  repaet run eval/significance_stats/significance_per_seed.csv
  repaet run eval/significance_stats/mean_class_delta_f1.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


NUM_CLASSES = 241


def read_predictions(eval_root: Path, method: str, seed: int) -> pd.DataFrame:
    prefix = "best_baseseed" if method == "baseline" else "best_fullseed"
    path = eval_root / f"{prefix}{seed}" / "predictions.csv"
    df = pd.read_csv(path)
    df["top5_hit"] = [
        int(int(y) in {int(x) for x in str(ids).split(";")})
        for y, ids in zip(df["true_id"], df["top5_ids"])
    ]
    return df


def read_class_f1(eval_root: Path, method: str, seed: int) -> np.ndarray:
    prefix = "best_baseseed" if method == "baseline" else "best_fullseed"
    path = eval_root / f"{prefix}{seed}" / "classification_report.csv"
    df = pd.read_csv(path, index_col=0)
    values = []
    for i in range(NUM_CLASSES):
        key = f"{i:03d}"
        values.append(float(df.loc[key, "f1-score"]))
    return np.asarray(values, dtype=float)


def f1_scores_fast(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    support = np.bincount(y_true, minlength=NUM_CLASSES).astype(float)
    pred_count = np.bincount(pred, minlength=NUM_CLASSES).astype(float)
    correct = pred == y_true
    tp = np.bincount(y_true[correct], minlength=NUM_CLASSES).astype(float)
    fp = pred_count - tp
    fn = support - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.sum(f1 * support) / np.sum(support))
    return macro_f1, weighted_f1


def paired_delta(y_true, base_pred, full_pred, base_top5, full_top5) -> dict:
    base_macro, base_weighted = f1_scores_fast(y_true, base_pred)
    full_macro, full_weighted = f1_scores_fast(y_true, full_pred)
    return {
        "top1": float(np.mean(full_pred == y_true) - np.mean(base_pred == y_true)),
        "top5": float(np.mean(full_top5) - np.mean(base_top5)),
        "macro_f1": float(full_macro - base_macro),
        "weighted_f1": float(full_weighted - base_weighted),
    }


def paired_bootstrap_ci(y_true, base_pred, full_pred, base_top5, full_top5, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = {k: [] for k in ["top1", "top5", "macro_f1", "weighted_f1"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d = paired_delta(
            y_true[idx],
            base_pred[idx],
            full_pred[idx],
            base_top5[idx],
            full_top5[idx],
        )
        for k, v in d.items():
            samples[k].append(v)
    return {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in samples.items()}


def mcnemar_exact(y_true, base_pred, full_pred) -> dict:
    base_correct = base_pred == y_true
    full_correct = full_pred == y_true
    b01 = int(np.sum((~base_correct) & full_correct))
    b10 = int(np.sum(base_correct & (~full_correct)))
    n = b01 + b10
    p = float(binomtest(min(b01, b10), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0
    return {
        "baseline_wrong_full_correct": b01,
        "baseline_correct_full_wrong": b10,
        "discordant": n,
        "p_value": p,
    }


def fmt_p(p: float) -> str:
    return f"{p:.3e}" if p < 1e-4 else f"{p:.6f}"


def fmt_pp(x: float) -> str:
    return f"{x * 100:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="repaet run eval")
    parser.add_argument("--seeds", default="1,25,42,50,100")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_root = (root / args.eval_root).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else eval_root / "significance_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    pooled_true = []
    pooled_base_pred = []
    pooled_full_pred = []
    pooled_base_top5 = []
    pooled_full_top5 = []
    per_seed_rows = []
    class_delta_f1 = []

    for seed in seeds:
        base = read_predictions(eval_root, "baseline", seed)
        full = read_predictions(eval_root, "full", seed)
        if len(base) != len(full):
            raise ValueError(f"Prediction length mismatch for seed {seed}: {len(base)} vs {len(full)}")
        if not (base["path"].to_numpy() == full["path"].to_numpy()).all():
            raise ValueError(f"Prediction path mismatch for seed {seed}")
        if not (base["true_id"].to_numpy() == full["true_id"].to_numpy()).all():
            raise ValueError(f"Prediction label mismatch for seed {seed}")

        y = base["true_id"].to_numpy(dtype=int)
        bp = base["pred_id"].to_numpy(dtype=int)
        fp = full["pred_id"].to_numpy(dtype=int)
        b5 = base["top5_hit"].to_numpy(dtype=int)
        f5 = full["top5_hit"].to_numpy(dtype=int)

        delta = paired_delta(y, bp, fp, b5, f5)
        mc = mcnemar_exact(y, bp, fp)
        base_f1 = read_class_f1(eval_root, "baseline", seed)
        full_f1 = read_class_f1(eval_root, "full", seed)
        diff_f1 = full_f1 - base_f1
        wx = wilcoxon(diff_f1, alternative="greater", zero_method="wilcox", correction=False, method="auto")

        per_seed_rows.append({
            "seed": seed,
            "n_images": len(y),
            **{f"delta_{k}": v for k, v in delta.items()},
            "mcnemar_b01": mc["baseline_wrong_full_correct"],
            "mcnemar_b10": mc["baseline_correct_full_wrong"],
            "mcnemar_p": mc["p_value"],
            "wilcoxon_W": float(wx.statistic),
            "wilcoxon_p_greater": float(wx.pvalue),
            "mean_delta_f1": float(np.mean(diff_f1)),
            "positive_classes": int(np.sum(diff_f1 > 0)),
            "negative_classes": int(np.sum(diff_f1 < 0)),
            "zero_classes": int(np.sum(diff_f1 == 0)),
        })

        pooled_true.append(y)
        pooled_base_pred.append(bp)
        pooled_full_pred.append(fp)
        pooled_base_top5.append(b5)
        pooled_full_top5.append(f5)
        class_delta_f1.append(diff_f1)

    y = np.concatenate(pooled_true)
    bp = np.concatenate(pooled_base_pred)
    fp = np.concatenate(pooled_full_pred)
    b5 = np.concatenate(pooled_base_top5)
    f5 = np.concatenate(pooled_full_top5)

    delta = paired_delta(y, bp, fp, b5, f5)
    ci = paired_bootstrap_ci(y, bp, fp, b5, f5, n_boot=args.n_boot, seed=20260617)
    mc = mcnemar_exact(y, bp, fp)
    mean_class_delta = np.mean(np.vstack(class_delta_f1), axis=0)
    wx = wilcoxon(mean_class_delta, alternative="greater", zero_method="wilcox", correction=False, method="auto")

    summary = {
        "comparison": "Full model vs Baseline",
        "seeds": ",".join(map(str, seeds)),
        "n_seed_image_pairs": int(len(y)),
        "mcnemar_b01_baseline_wrong_full_correct": mc["baseline_wrong_full_correct"],
        "mcnemar_b10_baseline_correct_full_wrong": mc["baseline_correct_full_wrong"],
        "mcnemar_discordant": mc["discordant"],
        "mcnemar_p_two_sided_exact": mc["p_value"],
        "delta_top1": delta["top1"],
        "delta_top1_ci_low": ci["top1"][0],
        "delta_top1_ci_high": ci["top1"][1],
        "delta_top5": delta["top5"],
        "delta_top5_ci_low": ci["top5"][0],
        "delta_top5_ci_high": ci["top5"][1],
        "delta_macro_f1": delta["macro_f1"],
        "delta_macro_f1_ci_low": ci["macro_f1"][0],
        "delta_macro_f1_ci_high": ci["macro_f1"][1],
        "delta_weighted_f1": delta["weighted_f1"],
        "delta_weighted_f1_ci_low": ci["weighted_f1"][0],
        "delta_weighted_f1_ci_high": ci["weighted_f1"][1],
        "wilcoxon_W": float(wx.statistic),
        "wilcoxon_p_greater": float(wx.pvalue),
        "wilcoxon_mean_delta_f1": float(np.mean(mean_class_delta)),
        "wilcoxon_positive_classes": int(np.sum(mean_class_delta > 0)),
        "wilcoxon_negative_classes": int(np.sum(mean_class_delta < 0)),
        "wilcoxon_zero_classes": int(np.sum(mean_class_delta == 0)),
    }

    pd.DataFrame([summary]).to_csv(out_dir / "significance_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_seed_rows).to_csv(out_dir / "significance_per_seed.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "class_id": np.arange(NUM_CLASSES),
        "mean_delta_f1": mean_class_delta,
    }).to_csv(out_dir / "mean_class_delta_f1.csv", index=False, encoding="utf-8-sig")
    (out_dir / "significance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "| Analysis | Statistic | Result |",
        "|---|---:|---:|",
        f"| McNemar's test | b01 / b10 | {mc['baseline_wrong_full_correct']} / {mc['baseline_correct_full_wrong']} |",
        f"| McNemar's test | two-sided exact p | {fmt_p(mc['p_value'])} |",
        f"| Paired bootstrap | Delta Top-1 | {fmt_pp(delta['top1'])} pp, 95% CI [{fmt_pp(ci['top1'][0])}, {fmt_pp(ci['top1'][1])}] |",
        f"| Paired bootstrap | Delta Top-5 | {fmt_pp(delta['top5'])} pp, 95% CI [{fmt_pp(ci['top5'][0])}, {fmt_pp(ci['top5'][1])}] |",
        f"| Paired bootstrap | Delta Macro-F1 | {fmt_pp(delta['macro_f1'])} pp, 95% CI [{fmt_pp(ci['macro_f1'][0])}, {fmt_pp(ci['macro_f1'][1])}] |",
        f"| Paired bootstrap | Delta Weighted-F1 | {fmt_pp(delta['weighted_f1'])} pp, 95% CI [{fmt_pp(ci['weighted_f1'][0])}, {fmt_pp(ci['weighted_f1'][1])}] |",
        f"| Wilcoxon signed-rank | mean class Delta F1 | {fmt_pp(summary['wilcoxon_mean_delta_f1'])} pp |",
        f"| Wilcoxon signed-rank | positive / negative / zero classes | {summary['wilcoxon_positive_classes']} / {summary['wilcoxon_negative_classes']} / {summary['wilcoxon_zero_classes']} |",
        f"| Wilcoxon signed-rank | one-sided p, Delta F1 > 0 | {fmt_p(float(wx.pvalue))} |",
    ]
    (out_dir / "significance_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()
