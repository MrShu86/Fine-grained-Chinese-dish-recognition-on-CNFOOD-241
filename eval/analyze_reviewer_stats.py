# -*- coding: utf-8 -*-
"""
Reviewer-oriented statistical analysis for paired baseline/full predictions.

This script analyzes one paired baseline/full prediction file pair. It was used
for the original single-run reviewer statistics, full per-class F1 table,
improved/degraded category lists, and full confusion matrices.

Outputs:
  - summary_metrics.csv
  - paired_tests.json
  - bootstrap_ci.csv
  - all_241_per_class_f1.csv
  - degraded_categories.csv
  - improved_categories.csv
  - confusion_matrix_baseline.csv
  - confusion_matrix_full.csv
  - top_confusions_baseline.csv
  - top_confusions_full.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, wilcoxon
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def parse_topk(value):
    if pd.isna(value):
        return []
    return [int(x) for x in str(value).split(";") if str(x).strip() != ""]


def exact_mcnemar(b01, b10):
    """Exact two-sided McNemar test using binomial test on discordant pairs."""
    n = int(b01 + b10)
    if n == 0:
        return {"b01": int(b01), "b10": int(b10), "n_discordant": 0, "p_value": 1.0}
    p = binomtest(k=min(int(b01), int(b10)), n=n, p=0.5, alternative="two-sided").pvalue
    return {"b01": int(b01), "b10": int(b10), "n_discordant": n, "p_value": float(p)}


def mcnemar_midp_approx(b01, b10):
    """Continuity-corrected chi-square approximation for reporting alongside exact test."""
    denom = b01 + b10
    if denom == 0:
        return {"chi2_cc": 0.0, "p_value_cc": 1.0}
    stat = (abs(b01 - b10) - 1) ** 2 / denom
    return {"chi2_cc": float(stat), "p_value_cc": float(chi2.sf(stat, df=1))}


def metric_pack(y_true, y_pred, top5_lists):
    top1 = float(np.mean(y_true == y_pred))
    top5 = float(np.mean([int(t) in top5 for t, top5 in zip(y_true, top5_lists)]))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return {"top1": top1, "top5": top5, "macro_f1": macro, "weighted_f1": weighted}


def paired_bootstrap(y_true, base_pred, full_pred, base_top5, full_top5, labels, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rows = []
    metrics = ["top1", "top5", "macro_f1", "weighted_f1"]
    diffs = {m: np.empty(n_boot, dtype=np.float64) for m in metrics}

    y_true = np.asarray(y_true)
    base_pred = np.asarray(base_pred)
    full_pred = np.asarray(full_pred)
    base_top5 = np.asarray(base_top5, dtype=object)
    full_top5 = np.asarray(full_top5, dtype=object)
    base_top1_correct = (y_true == base_pred).astype(np.float64)
    full_top1_correct = (y_true == full_pred).astype(np.float64)
    base_top5_correct = np.array([int(t) in top5 for t, top5 in zip(y_true, base_top5)], dtype=np.float64)
    full_top5_correct = np.array([int(t) in top5 for t, top5 in zip(y_true, full_top5)], dtype=np.float64)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs["top1"][i] = float(full_top1_correct[idx].mean() - base_top1_correct[idx].mean())
        diffs["top5"][i] = float(full_top5_correct[idx].mean() - base_top5_correct[idx].mean())
        diffs["macro_f1"][i] = float(
            f1_score(y_true[idx], full_pred[idx], labels=labels, average="macro", zero_division=0)
            - f1_score(y_true[idx], base_pred[idx], labels=labels, average="macro", zero_division=0)
        )
        diffs["weighted_f1"][i] = float(
            f1_score(y_true[idx], full_pred[idx], labels=labels, average="weighted", zero_division=0)
            - f1_score(y_true[idx], base_pred[idx], labels=labels, average="weighted", zero_division=0)
        )

    for m in metrics:
        vals = diffs[m]
        rows.append({
            "metric": m,
            "delta_mean": float(np.mean(vals)),
            "ci95_low": float(np.percentile(vals, 2.5)),
            "ci95_high": float(np.percentile(vals, 97.5)),
            "n_boot": int(n_boot),
            "seed": int(seed),
        })
    return pd.DataFrame(rows)


def per_class_table(y_true, base_pred, full_pred, labels):
    p_b, r_b, f_b, s = precision_recall_fscore_support(
        y_true, base_pred, labels=labels, zero_division=0
    )
    p_f, r_f, f_f, _ = precision_recall_fscore_support(
        y_true, full_pred, labels=labels, zero_division=0
    )
    df = pd.DataFrame({
        "class_id": labels,
        "class_label": [f"{i:03d}" for i in labels],
        "support": s,
        "baseline_precision": p_b,
        "baseline_recall": r_b,
        "baseline_f1": f_b,
        "full_precision": p_f,
        "full_recall": r_f,
        "full_f1": f_f,
    })
    df["delta_f1"] = df["full_f1"] - df["baseline_f1"]
    return df


def top_confusions(cm, labels, k=50):
    rows = []
    row_sum = cm.sum(axis=1)
    for i, true_id in enumerate(labels):
        total = max(int(row_sum[i]), 1)
        for j, pred_id in enumerate(labels):
            if true_id == pred_id:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            rows.append({
                "true_id": int(true_id),
                "true_label": f"{true_id:03d}",
                "pred_id": int(pred_id),
                "pred_label": f"{pred_id:03d}",
                "count": count,
                "true_class_total": total,
                "rate_in_true": count / total,
            })
    return pd.DataFrame(rows).sort_values(["count", "rate_in_true"], ascending=False).head(k)


def add_main_confusion_delta(per_cls, cm_base, cm_full, labels):
    label_to_idx = {v: i for i, v in enumerate(labels)}
    rows = []
    for _, row in per_cls.iterrows():
        class_id = int(row["class_id"])
        i = label_to_idx[class_id]
        base_row = cm_base[i].copy()
        full_row = cm_full[i].copy()
        base_row[i] = 0
        full_row[i] = 0
        base_pred = int(labels[int(np.argmax(base_row))])
        full_pred = int(labels[int(np.argmax(full_row))])
        rows.append({
            "class_id": class_id,
            "baseline_main_confused_with": base_pred,
            "baseline_main_confusion_count": int(base_row.max()),
            "full_main_confused_with": full_pred,
            "full_main_confusion_count": int(full_row.max()),
        })
    return per_cls.merge(pd.DataFrame(rows), on="class_id", how="left")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="baseline_predictions.csv")
    parser.add_argument("--full", default="full_predictions.csv")
    parser.add_argument("--out-dir", default="reviewer_stats_outputs")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = pd.read_csv(args.baseline)
    full = pd.read_csv(args.full)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = {"path", "true_id", "pred_id", "top5_ids"}
    missing_base = required - set(base.columns)
    missing_full = required - set(full.columns)
    if missing_base or missing_full:
        raise ValueError(f"Missing columns: baseline={missing_base}, full={missing_full}")

    merged = base.merge(
        full,
        on=["path", "true_id"],
        suffixes=("_baseline", "_full"),
        validate="one_to_one",
    ).sort_values("path").reset_index(drop=True)

    y_true = merged["true_id"].to_numpy(dtype=int)
    yb = merged["pred_id_baseline"].to_numpy(dtype=int)
    yf = merged["pred_id_full"].to_numpy(dtype=int)
    top5_b = merged["top5_ids_baseline"].map(parse_topk).to_numpy(dtype=object)
    top5_f = merged["top5_ids_full"].map(parse_topk).to_numpy(dtype=object)
    labels = np.array(sorted(pd.unique(y_true)), dtype=int)

    base_metrics = metric_pack(y_true, yb, top5_b)
    full_metrics = metric_pack(y_true, yf, top5_f)
    summary = pd.DataFrame([
        {"model": "baseline", **base_metrics},
        {"model": "full", **full_metrics},
        {"model": "delta_full_minus_baseline", **{k: full_metrics[k] - base_metrics[k] for k in base_metrics}},
    ])
    summary.to_csv(out_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")

    base_correct = y_true == yb
    full_correct = y_true == yf
    b01 = int(np.sum((~base_correct) & full_correct))
    b10 = int(np.sum(base_correct & (~full_correct)))

    per_cls = per_class_table(y_true, yb, yf, labels)
    cm_b = confusion_matrix(y_true, yb, labels=labels)
    cm_f = confusion_matrix(y_true, yf, labels=labels)
    per_cls = add_main_confusion_delta(per_cls, cm_b, cm_f, labels)
    per_cls.to_csv(out_dir / "all_241_per_class_f1.csv", index=False, encoding="utf-8-sig")
    per_cls[per_cls["delta_f1"] < 0].sort_values("delta_f1").to_csv(
        out_dir / "degraded_categories.csv", index=False, encoding="utf-8-sig"
    )
    per_cls[per_cls["delta_f1"] > 0].sort_values("delta_f1", ascending=False).to_csv(
        out_dir / "improved_categories.csv", index=False, encoding="utf-8-sig"
    )

    bootstrap = paired_bootstrap(
        y_true, yb, yf, top5_b, top5_f, labels=labels, n_boot=args.n_boot, seed=args.seed
    )
    bootstrap.to_csv(out_dir / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")

    try:
        w = wilcoxon(per_cls["delta_f1"].to_numpy(), alternative="greater", zero_method="wilcox")
        wilcoxon_result = {"statistic": float(w.statistic), "p_value_greater_than_zero": float(w.pvalue)}
    except ValueError as exc:
        wilcoxon_result = {"error": str(exc)}

    tests = {
        "n_samples": int(len(merged)),
        "n_classes": int(len(labels)),
        "mcnemar_exact_top1": exact_mcnemar(b01=b01, b10=b10),
        "mcnemar_chi_square_cc_top1": mcnemar_midp_approx(b01=b01, b10=b10),
        "wilcoxon_delta_f1_greater_than_zero": wilcoxon_result,
        "delta_f1_summary": {
            "improved_classes": int((per_cls["delta_f1"] > 0).sum()),
            "degraded_classes": int((per_cls["delta_f1"] < 0).sum()),
            "unchanged_classes": int((per_cls["delta_f1"] == 0).sum()),
            "mean": float(per_cls["delta_f1"].mean()),
            "median": float(per_cls["delta_f1"].median()),
            "std": float(per_cls["delta_f1"].std(ddof=1)),
            "q1": float(per_cls["delta_f1"].quantile(0.25)),
            "q3": float(per_cls["delta_f1"].quantile(0.75)),
        },
    }
    (out_dir / "paired_tests.json").write_text(
        json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.DataFrame(cm_b, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix_baseline.csv", encoding="utf-8-sig")
    pd.DataFrame(cm_f, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix_full.csv", encoding="utf-8-sig")
    top_confusions(cm_b, labels).to_csv(out_dir / "top_confusions_baseline.csv", index=False, encoding="utf-8-sig")
    top_confusions(cm_f, labels).to_csv(out_dir / "top_confusions_full.csv", index=False, encoding="utf-8-sig")

    print("Saved outputs to:", out_dir.resolve())
    print(summary.to_string(index=False))
    print(json.dumps(tests, ensure_ascii=False, indent=2))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
