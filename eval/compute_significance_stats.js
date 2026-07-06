const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const evalRoot = path.join(root, "repaet run eval");
const outDir = path.join(evalRoot, "significance_stats");
fs.mkdirSync(outDir, { recursive: true });

const seeds = [1, 25, 42, 50, 100];
const NUM_CLASSES = 241;
const BOOT = 2000;

function parseCsvLine(line) {
  const out = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQ && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else inQ = !inQ;
    } else if (ch === "," && !inQ) {
      out.push(cur);
      cur = "";
    } else cur += ch;
  }
  out.push(cur);
  return out;
}

function readCsv(file) {
  const text = fs.readFileSync(file, "utf8").replace(/^\uFEFF/, "").trim();
  const lines = text.split(/\r?\n/).filter(Boolean);
  const header = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const vals = parseCsvLine(line);
    const obj = {};
    header.forEach((h, i) => {
      obj[h] = vals[i];
    });
    return obj;
  });
}

function readPred(method, seed) {
  const dir = path.join(evalRoot, method === "baseline" ? `best_baseseed${seed}` : `best_fullseed${seed}`);
  return readCsv(path.join(dir, "predictions.csv")).map((r) => ({
    path: r.path,
    y: Number(r.true_id),
    pred: Number(r.pred_id),
    top5: r.top5_ids.split(";").map(Number),
  }));
}

function readReport(method, seed) {
  const dir = path.join(evalRoot, method === "baseline" ? `best_baseseed${seed}` : `best_fullseed${seed}`);
  const rows = readCsv(path.join(dir, "classification_report.csv"));
  const f1 = new Map();
  for (const r of rows) {
    const label = r[""];
    if (/^\d+$/.test(label)) f1.set(Number(label), Number(r["f1-score"]));
  }
  return f1;
}

function metricsFromRecords(records) {
  const n = records.length;
  let top1 = 0;
  let top5 = 0;
  const tp = Array(NUM_CLASSES).fill(0);
  const fp = Array(NUM_CLASSES).fill(0);
  const fn = Array(NUM_CLASSES).fill(0);
  const support = Array(NUM_CLASSES).fill(0);
  for (const r of records) {
    if (r.pred === r.y) top1++;
    if (r.top5.includes(r.y)) top5++;
    support[r.y]++;
    if (r.pred === r.y) tp[r.y]++;
    else {
      fp[r.pred]++;
      fn[r.y]++;
    }
  }
  let macro = 0;
  let weighted = 0;
  let totalSupport = 0;
  for (let c = 0; c < NUM_CLASSES; c++) {
    const denom = 2 * tp[c] + fp[c] + fn[c];
    const f1 = denom > 0 ? (2 * tp[c]) / denom : 0;
    macro += f1;
    weighted += f1 * support[c];
    totalSupport += support[c];
  }
  return {
    top1: top1 / n,
    top5: top5 / n,
    macro_f1: macro / NUM_CLASSES,
    weighted_f1: weighted / totalSupport,
  };
}

function pairedMetrics(base, full, indices = null) {
  const b = [];
  const f = [];
  const idxs = indices || base.map((_, i) => i);
  for (const i of idxs) {
    b.push(base[i]);
    f.push(full[i]);
  }
  const mb = metricsFromRecords(b);
  const mf = metricsFromRecords(f);
  return {
    top1: mf.top1 - mb.top1,
    top5: mf.top5 - mb.top5,
    macro_f1: mf.macro_f1 - mb.macro_f1,
    weighted_f1: mf.weighted_f1 - mb.weighted_f1,
  };
}

function mulberry32(a) {
  return function rng() {
    let t = (a += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function percentile(arr, p) {
  const a = [...arr].sort((x, y) => x - y);
  const pos = (a.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return a[lo];
  return a[lo] + (a[hi] - a[lo]) * (pos - lo);
}

function bootstrapCI(base, full, reps = BOOT, seed = 12345) {
  const rng = mulberry32(seed);
  const n = base.length;
  const vals = { top1: [], top5: [], macro_f1: [], weighted_f1: [] };
  for (let r = 0; r < reps; r++) {
    const idx = Array(n);
    for (let i = 0; i < n; i++) idx[i] = Math.floor(rng() * n);
    const d = pairedMetrics(base, full, idx);
    for (const k of Object.keys(vals)) vals[k].push(d[k]);
  }
  const out = {};
  for (const k of Object.keys(vals)) out[k] = [percentile(vals[k], 0.025), percentile(vals[k], 0.975)];
  return out;
}

function logChoose(n, k) {
  if (k < 0 || k > n) return -Infinity;
  k = Math.min(k, n - k);
  let s = 0;
  for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i);
  return s;
}

function binomCdf(k, n, p = 0.5) {
  if (k < 0) return 0;
  if (k >= n) return 1;
  const logs = [];
  let max = -Infinity;
  for (let i = 0; i <= k; i++) {
    const v = logChoose(n, i) + i * Math.log(p) + (n - i) * Math.log(1 - p);
    logs.push(v);
    if (v > max) max = v;
  }
  let sum = 0;
  for (const v of logs) sum += Math.exp(v - max);
  return Math.exp(max) * sum;
}

function mcnemar(base, full) {
  let b01 = 0;
  let b10 = 0;
  for (let i = 0; i < base.length; i++) {
    const bc = base[i].pred === base[i].y;
    const fc = full[i].pred === full[i].y;
    if (!bc && fc) b01++;
    if (bc && !fc) b10++;
  }
  const n = b01 + b10;
  const p = Math.min(1, 2 * binomCdf(Math.min(b01, b10), n, 0.5));
  return { baseline_wrong_full_correct: b01, baseline_correct_full_wrong: b10, discordant: n, p_value: p };
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x));
  return sign * y;
}

function normCdf(z) {
  return 0.5 * (1 + erf(z / Math.SQRT2));
}

function wilcoxonGreater(diffs) {
  const nz = diffs.filter((x) => Math.abs(x) > 1e-15);
  const arr = nz.map((x) => ({ abs: Math.abs(x), sign: Math.sign(x) })).sort((a, b) => a.abs - b.abs);
  let W = 0;
  const tieCounts = [];
  for (let i = 0; i < arr.length;) {
    let j = i + 1;
    while (j < arr.length && Math.abs(arr[j].abs - arr[i].abs) < 1e-15) j++;
    const avg = (i + 1 + j) / 2;
    const count = j - i;
    tieCounts.push(count);
    for (let k = i; k < j; k++) if (arr[k].sign > 0) W += avg;
    i = j;
  }
  const n = arr.length;
  const mean = (n * (n + 1)) / 4;
  let variance = (n * (n + 1) * (2 * n + 1)) / 24;
  let tieAdj = 0;
  for (const t of tieCounts) tieAdj += t * t * t - t;
  variance -= tieAdj / 48;
  const z = (W - mean - 0.5) / Math.sqrt(variance);
  const p = 1 - normCdf(z);
  return {
    n_nonzero: n,
    W_plus: W,
    z,
    p_value_greater: p,
    mean_delta: diffs.reduce((a, b) => a + b, 0) / diffs.length,
    positives: diffs.filter((x) => x > 0).length,
    negatives: diffs.filter((x) => x < 0).length,
    zeros: diffs.filter((x) => Math.abs(x) <= 1e-15).length,
  };
}

function csvEscape(x) {
  const s = String(x);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function writeCsv(file, rows) {
  const keys = Object.keys(rows[0]);
  const lines = [keys.join(",")];
  for (const r of rows) lines.push(keys.map((k) => csvEscape(r[k])).join(","));
  fs.writeFileSync(file, lines.join("\n"), "utf8");
}

function fmtPct(x) {
  return (x * 100).toFixed(3);
}

function fmtP(p) {
  return p < 1e-4 ? p.toExponential(3) : p.toFixed(6);
}

const perSeedRows = [];
const pooledBase = [];
const pooledFull = [];

for (const seed of seeds) {
  const b = readPred("baseline", seed);
  const f = readPred("full", seed);
  if (b.length !== f.length) throw new Error(`length mismatch seed ${seed}`);
  for (let i = 0; i < b.length; i++) {
    if (b[i].path !== f[i].path || b[i].y !== f[i].y) throw new Error(`pair mismatch seed ${seed} idx ${i}`);
  }
  pooledBase.push(...b);
  pooledFull.push(...f);
  const d = pairedMetrics(b, f);
  const ci = bootstrapCI(b, f, BOOT, 1000 + seed);
  const mc = mcnemar(b, f);
  const rb = readReport("baseline", seed);
  const rf = readReport("full", seed);
  const diffs = [];
  for (let c = 0; c < NUM_CLASSES; c++) diffs.push((rf.get(c) || 0) - (rb.get(c) || 0));
  const wx = wilcoxonGreater(diffs);
  perSeedRows.push({
    seed,
    n: b.length,
    delta_top1: d.top1,
    delta_top5: d.top5,
    delta_macro_f1: d.macro_f1,
    delta_weighted_f1: d.weighted_f1,
    top1_ci_low: ci.top1[0],
    top1_ci_high: ci.top1[1],
    top5_ci_low: ci.top5[0],
    top5_ci_high: ci.top5[1],
    macro_f1_ci_low: ci.macro_f1[0],
    macro_f1_ci_high: ci.macro_f1[1],
    weighted_f1_ci_low: ci.weighted_f1[0],
    weighted_f1_ci_high: ci.weighted_f1[1],
    mcnemar_b01: mc.baseline_wrong_full_correct,
    mcnemar_b10: mc.baseline_correct_full_wrong,
    mcnemar_p: mc.p_value,
    wilcoxon_W_plus: wx.W_plus,
    wilcoxon_z: wx.z,
    wilcoxon_p_greater: wx.p_value_greater,
    wilcoxon_mean_delta_f1: wx.mean_delta,
    wilcoxon_pos: wx.positives,
    wilcoxon_neg: wx.negatives,
    wilcoxon_zero: wx.zeros,
  });
}

const pooledD = pairedMetrics(pooledBase, pooledFull);
const pooledCI = bootstrapCI(pooledBase, pooledFull, BOOT, 999);
const pooledMC = mcnemar(pooledBase, pooledFull);
const meanClassDiffs = [];
for (let c = 0; c < NUM_CLASSES; c++) {
  let s = 0;
  for (const seed of seeds) {
    const rb = readReport("baseline", seed);
    const rf = readReport("full", seed);
    s += (rf.get(c) || 0) - (rb.get(c) || 0);
  }
  meanClassDiffs.push(s / seeds.length);
}
const pooledWX = wilcoxonGreater(meanClassDiffs);

writeCsv(path.join(outDir, "significance_per_seed.csv"), perSeedRows);
writeCsv(path.join(outDir, "mean_class_delta_f1.csv"), meanClassDiffs.map((d, i) => ({ class_id: i, mean_delta_f1: d })));

const summaryRows = [{
  comparison: "Full model vs Baseline (pooled seed-image pairs)",
  seeds: seeds.join(";"),
  n_pairs: pooledBase.length,
  mcnemar_b01: pooledMC.baseline_wrong_full_correct,
  mcnemar_b10: pooledMC.baseline_correct_full_wrong,
  mcnemar_discordant: pooledMC.discordant,
  mcnemar_p_value: pooledMC.p_value,
  delta_top1: pooledD.top1,
  delta_top1_ci: `[${pooledCI.top1[0]}, ${pooledCI.top1[1]}]`,
  delta_top5: pooledD.top5,
  delta_top5_ci: `[${pooledCI.top5[0]}, ${pooledCI.top5[1]}]`,
  delta_macro_f1: pooledD.macro_f1,
  delta_macro_f1_ci: `[${pooledCI.macro_f1[0]}, ${pooledCI.macro_f1[1]}]`,
  delta_weighted_f1: pooledD.weighted_f1,
  delta_weighted_f1_ci: `[${pooledCI.weighted_f1[0]}, ${pooledCI.weighted_f1[1]}]`,
  wilcoxon_n_classes_nonzero: pooledWX.n_nonzero,
  wilcoxon_W_plus: pooledWX.W_plus,
  wilcoxon_z: pooledWX.z,
  wilcoxon_p_greater: pooledWX.p_value_greater,
  wilcoxon_mean_delta_f1: pooledWX.mean_delta,
  wilcoxon_pos_classes: pooledWX.positives,
  wilcoxon_neg_classes: pooledWX.negatives,
  wilcoxon_zero_classes: pooledWX.zeros,
}];
writeCsv(path.join(outDir, "significance_summary.csv"), summaryRows);

const md = [];
md.push("| Analysis | Statistic | Result |");
md.push("|---|---:|---:|");
md.push(`| McNemar's test | b01 / b10 | ${pooledMC.baseline_wrong_full_correct} / ${pooledMC.baseline_correct_full_wrong} |`);
md.push(`| McNemar's test | two-sided exact p | ${fmtP(pooledMC.p_value)} |`);
md.push(`| Paired bootstrap | ΔTop-1 | ${fmtPct(pooledD.top1)} pp, 95% CI [${fmtPct(pooledCI.top1[0])}, ${fmtPct(pooledCI.top1[1])}] |`);
md.push(`| Paired bootstrap | ΔTop-5 | ${fmtPct(pooledD.top5)} pp, 95% CI [${fmtPct(pooledCI.top5[0])}, ${fmtPct(pooledCI.top5[1])}] |`);
md.push(`| Paired bootstrap | ΔMacro-F1 | ${fmtPct(pooledD.macro_f1)} pp, 95% CI [${fmtPct(pooledCI.macro_f1[0])}, ${fmtPct(pooledCI.macro_f1[1])}] |`);
md.push(`| Paired bootstrap | ΔWeighted-F1 | ${fmtPct(pooledD.weighted_f1)} pp, 95% CI [${fmtPct(pooledCI.weighted_f1[0])}, ${fmtPct(pooledCI.weighted_f1[1])}] |`);
md.push(`| Wilcoxon signed-rank | mean class ΔF1 | ${fmtPct(pooledWX.mean_delta)} pp |`);
md.push(`| Wilcoxon signed-rank | positive / negative / zero classes | ${pooledWX.positives} / ${pooledWX.negatives} / ${pooledWX.zeros} |`);
md.push(`| Wilcoxon signed-rank | one-sided p, ΔF1 > 0 | ${fmtP(pooledWX.p_value_greater)} |`);
fs.writeFileSync(path.join(outDir, "significance_summary.md"), md.join("\n"), "utf8");

console.log(md.join("\n"));
console.log(`Saved to ${outDir}`);
