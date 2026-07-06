# -*- coding: utf-8 -*-
"""
Run R_KDAT ablation experiments for reviewer-requested component synergy.

Variants:
  kd_mbc : KD + MBC(ArcFace), no ER(Triplet), no strong augmentation
  kd_er  : KD + ER(Triplet), no MBC(ArcFace), no strong augmentation
  mbc_er : MBC(ArcFace) + ER(Triplet), no KD, no strong augmentation
  core   : KD + MBC(ArcFace) + ER(Triplet), no strong augmentation
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "R_KDAT"

VARIANTS = {
    "kd_mbc": {
        "title": "KD + MBC",
        "flags": ["--kd-use", "--arcface", "--no-triplet", "--no-strong-aug"],
        "needs_teacher": True,
    },
    "kd_er": {
        "title": "KD + ER",
        "flags": ["--kd-use", "--no-arcface", "--triplet", "--no-strong-aug"],
        "needs_teacher": True,
    },
    "mbc_er": {
        "title": "MBC + ER",
        "flags": ["--no-kd", "--arcface", "--triplet", "--no-strong-aug"],
        "needs_teacher": False,
    },
    "core": {
        "title": "KD + MBC + ER",
        "flags": ["--kd-use", "--arcface", "--triplet", "--no-strong-aug"],
        "needs_teacher": True,
    },
}


def split_csv(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def split_seeds(text):
    return [int(x) for x in split_csv(text)]


def resolve_teacher_ckpt(args, seed):
    if args.teacher_root:
        return str(Path(args.teacher_root) / "baseline" / f"seed_{seed}" / "best.pt")
    return args.teacher_ckpt


def build_command(args, variant, seed):
    spec = VARIANTS[variant]
    out_dir = Path(args.out_root) / variant / f"seed_{seed}"
    cmd = [
        args.python,
        str(MODEL_DIR / "main.py"),
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
        ("--pk-p", args.pk_p),
        ("--pk-k", args.pk_k),
        ("--amp-dtype", args.amp_dtype),
    ]
    for flag, value in optional:
        if value is not None:
            cmd.extend([flag, str(value)])

    if spec["needs_teacher"]:
        teacher_ckpt = resolve_teacher_ckpt(args, seed)
        if teacher_ckpt:
            cmd.extend(["--teacher-ckpt", teacher_ckpt])

    cmd.extend(spec["flags"])

    if args.no_tensorboard:
        cmd.append("--no-tensorboard")
    if args.deterministic:
        cmd.append("--deterministic")
    if args.no_channels_last:
        cmd.append("--no-channels-last")
    if args.no_fused_adamw:
        cmd.append("--no-fused-adamw")
    if args.compile:
        cmd.append("--compile")

    return cmd, out_dir


def run_one(args, variant, seed):
    cmd, out_dir = build_command(args, variant, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "experiment": "R_KDAT_ablation",
        "variant": variant,
        "title": VARIANTS[variant]["title"],
        "seed": seed,
        "out_dir": str(out_dir),
        "command": cmd,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[RUN] variant={variant} ({VARIANTS[variant]['title']}) seed={seed}")
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
        raise SystemExit(f"Run failed: variant={variant} seed={seed} return_code={ret}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run R_KDAT component ablation experiments.")
    parser.add_argument("--variants", default="kd_mbc,kd_er,mbc_er,core",
                        help="Comma-separated variants: kd_mbc,kd_er,mbc_er,core")
    parser.add_argument("--seeds", default="1,25,50,100,42")
    parser.add_argument("--out-root", default=str(ROOT / "ablation_runs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--val-dir", default=None)
    parser.add_argument("--teacher-ckpt", default=None,
                        help="Use the same baseline teacher checkpoint for all KD variants.")
    parser.add_argument("--teacher-root", default=None,
                        help="Use seed-matched teachers from <teacher_root>/baseline/seed_<seed>/best.pt.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pk-p", type=int, default=None)
    parser.add_argument("--pk-k", type=int, default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "off"], default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-channels-last", action="store_true")
    parser.add_argument("--no-fused-adamw", action="store_true")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    variants = split_csv(args.variants)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}. Choose from {list(VARIANTS)}")

    seeds = split_seeds(args.seeds)
    print("[INFO] variants:", variants)
    print("[INFO] seeds:", seeds)
    print("[INFO] out_root:", args.out_root)

    for variant in variants:
        for seed in seeds:
            run_one(args, variant, seed)

    print("\n[DONE] all ablation experiments finished.")


if __name__ == "__main__":
    main()
