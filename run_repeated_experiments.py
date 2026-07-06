# -*- coding: utf-8 -*-
"""
Run repeated training experiments with multiple random seeds.

Examples:
  # Baseline, 5 seeds
  C:\Anaconda\envs\opencv_py38\python.exe run_repeated_experiments.py ^
    --model baseline ^
    --train-dir "C:\CNFOOD-241\CNFOOD-241\train600x600" ^
    --val-dir "C:\CNFOOD-241\CNFOOD-241\val600x600"

  # Full R_KDAT model, 5 seeds, using a baseline teacher checkpoint
  C:\Anaconda\envs\opencv_py38\python.exe run_repeated_experiments.py ^
    --model full ^
    --teacher-ckpt "C:\path\to\baseline_teacher\best.pt" ^
    --train-dir "C:\CNFOOD-241\CNFOOD-241\train600x600" ^
    --val-dir "C:\CNFOOD-241\CNFOOD-241\val600x600"
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIRS = {
    "baseline": ROOT / "RegNetY-32GF",
    "full": ROOT / "R_KDAT",
}


def split_seeds(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def resolve_teacher_ckpt(args, seed):
    if args.teacher_root:
        candidate = Path(args.teacher_root) / "baseline" / f"seed_{seed}" / "best.pt"
        return str(candidate)
    return args.teacher_ckpt


def build_command(args, model_name, seed):
    model_dir = MODEL_DIRS[model_name]
    out_dir = Path(args.out_root) / model_name / f"seed_{seed}"
    cmd = [
        args.python,
        str(model_dir / "main.py"),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--resume-ckpt", "",
    ]

    optional = [
        ("--train-dir", args.train_dir),
        ("--val-dir", args.val_dir),
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--num-workers", args.num_workers),
    ]
    for flag, value in optional:
        if value is not None:
            cmd.extend([flag, str(value)])

    if model_name == "full":
        teacher_ckpt = resolve_teacher_ckpt(args, seed)
        if teacher_ckpt:
            cmd.extend(["--teacher-ckpt", teacher_ckpt])
        if args.pk_p is not None:
            cmd.extend(["--pk-p", str(args.pk_p)])
        if args.pk_k is not None:
            cmd.extend(["--pk-k", str(args.pk_k)])

    if args.no_tensorboard:
        cmd.append("--no-tensorboard")
    if args.deterministic:
        cmd.append("--deterministic")
    if args.amp_dtype:
        cmd.extend(["--amp-dtype", args.amp_dtype])
    if args.no_channels_last:
        cmd.append("--no-channels-last")
    if args.no_fused_adamw:
        cmd.append("--no-fused-adamw")
    if args.compile:
        cmd.append("--compile")

    return cmd, out_dir


def run_one(args, model_name, seed):
    cmd, out_dir = build_command(args, model_name, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": model_name,
        "seed": seed,
        "out_dir": str(out_dir),
        "command": cmd,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[RUN] model={model_name} seed={seed}")
    print("[OUT]", out_dir)
    print("[CMD]", " ".join(f'"{x}"' if " " in x else x for x in cmd))

    with (out_dir / "train.log").open("w", encoding="utf-8", errors="replace") as log:
        log.write("[CMD] " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        ret = proc.wait()

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["return_code"] = ret
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if ret != 0:
        raise SystemExit(f"Run failed: model={model_name} seed={seed} return_code={ret}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-seed repeated experiments.")
    parser.add_argument("--model", choices=["baseline", "full", "both"], required=True)
    parser.add_argument("--seeds", default="42,1,100,25,50")
    parser.add_argument("--out-root", default=str(ROOT / "repeated_runs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--teacher-ckpt", default=None,
                        help="Use the same teacher checkpoint for all full-model seeds.")
    parser.add_argument("--teacher-root", default=None,
                        help="Use seed-matched teachers from <teacher_root>/baseline/seed_<seed>/best.pt.")
    parser.add_argument("--pk-p", type=int, default=None)
    parser.add_argument("--pk-k", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default=None)
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-fused-adamw", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = split_seeds(args.seeds)
    models = ["baseline", "full"] if args.model == "both" else [args.model]

    print("[INFO] seeds:", seeds)
    print("[INFO] models:", models)
    print("[INFO] out_root:", args.out_root)

    for model_name in models:
        for seed in seeds:
            run_one(args, model_name, seed)

    print("\n[DONE] all repeated experiments finished.")


if __name__ == "__main__":
    main()
