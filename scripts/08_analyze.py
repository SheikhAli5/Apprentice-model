"""Step 8: final table, paired tests and the figure from the run files in results/runs.
Only needs results/, no index and no training:  python scripts/08_analyze.py"""
import json, os
import numpy as np
import ir_measures
from ir_measures import nDCG, RR
from scipy.stats import ttest_rel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/runs"
TAGS = ["tiny", "mini", "small", "base"]
PARAMS = {"tiny": 4.4, "mini": 11.2, "small": 28.8, "base": 109.5}   # millions
SEEDS = (1, 2, 3)
SPLITS = ["dl19", "dl20"]
QRELS = "results/qrels.json"                     # labels only, small enough to keep in the repo
if not os.path.exists(QRELS):                    # made once from data/eval_*.json (step 6)
    with open(QRELS, "w") as f:
        json.dump({s: json.load(open(f"data/eval_{s}.json"))["qrels"] for s in SPLITS + ["val"]}, f)
qrels = json.load(open(QRELS))


def load(prefix):
    # -> ({qid: nDCG@10} over DL19+DL20, {qid: run}, val RR@10), or None if files are missing
    pq, runs = {}, {}
    for s in SPLITS:
        path = f"{prefix}_{s}.json"
        if not os.path.exists(path):
            return None
        run = json.load(open(path))
        runs.update(run)
        for m in ir_measures.iter_calc([nDCG @ 10], qrels[s], run):
            pq[m.query_id] = m.value
    val = None
    if os.path.exists(f"{prefix}_val.json"):
        val = ir_measures.calc_aggregate([RR @ 10], qrels["val"], json.load(open(f"{prefix}_val.json")))[RR @ 10]
    return pq, runs, val


def system(prefixes):
    # mean over seeds per query, keeps the single seeds too
    seeds = [x for x in (load(p) for p in prefixes) if x]
    if not seeds:
        return None
    qids = sorted(seeds[0][0])
    vec = np.mean([[pq[q] for q in qids] for pq, _, _ in seeds], axis=0)
    return {"qids": qids, "vec": vec, "seeds": seeds}


def split_mean(pq_or_sys, split):
    ids = set(qrels[split])
    if isinstance(pq_or_sys, dict) and "vec" in pq_or_sys:
        return float(np.mean([v for q, v in zip(pq_or_sys["qids"], pq_or_sys["vec"]) if q in ids]))
    return float(np.mean([v for q, v in pq_or_sys.items() if q in ids]))


def overlap10(runs, teacher_runs):
    # share of passages in both the student's and the teacher's top-10
    vals = []
    for q, run in runs.items():
        top_s = sorted(run, key=run.get, reverse=True)[:10]
        t = teacher_runs[q]
        top_t = sorted(t, key=t.get, reverse=True)[:10]
        vals.append(len(set(top_s) & set(top_t)) / 10)
    return float(np.mean(vals))


def holm(pvals):
    order = np.argsort(pvals)
    adj, m, running = np.empty(len(pvals)), len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# all systems
S = {"BM25": system([f"{R}/bm25"]),
     "Teacher": system([f"{R}/teacher"]),
     "Ref-MiniLM-L6 (500k labels)": system([f"{R}/ref_msmarco-MiniLM-L6"])}
for tag in TAGS:
    for cond in ["H1", "H", "HT", "T1", "T"]:
        S[f"{tag}-{cond}"] = system([f"{R}/student_{tag}_{cond}_s{s}" for s in SEEDS])
    for mode in ["H", "T"]:
        S[f"{tag}-{mode}nat"] = system([f"{R}/diag_{tag}_{mode}nat_s{s}" for s in SEEDS])
S = {k: v for k, v in S.items() if v}
teacher_runs = S["Teacher"]["seeds"][0][1]
t19, t20 = split_mean(S["Teacher"], "dl19"), split_mean(S["Teacher"], "dl20")

# table
print(f"{'system':28s} {'seeds':>5s} {'DL19':>13s} {'DL20':>13s} {'%T19':>6s} {'%T20':>6s} {'val RR':>7s} {'ovl@10':>7s}")
summary = {}
for name, sd in S.items():
    per19 = [split_mean(pq, "dl19") for pq, _, _ in sd["seeds"]]
    per20 = [split_mean(pq, "dl20") for pq, _, _ in sd["seeds"]]
    vals = [v for _, _, v in sd["seeds"] if v is not None]
    d19, d20 = float(np.mean(per19)), float(np.mean(per20))
    s19, s20 = float(np.std(per19)), float(np.std(per20))
    val = float(np.mean(vals)) if vals else float("nan")
    ovl = float(np.mean([overlap10(r, teacher_runs) for _, r, _ in sd["seeds"]]))
    summary[name] = {"seeds": len(sd["seeds"]), "dl19": d19, "dl19_std": s19, "dl20": d20, "dl20_std": s20,
                     "ret19": d19 / t19, "ret20": d20 / t20, "val_rr10": val, "overlap10": ovl}
    print(f"{name:28s} {len(sd['seeds']):5d} {d19:6.3f}±{s19:.3f} {d20:6.3f}±{s20:.3f} "
          f"{d19 / t19:6.1%} {d20 / t20:6.1%} {val:7.3f} {ovl:7.2f}")

# paired t-tests over the 97 queries (DL19+DL20), Holm correction over the sizes in each comparison
COMPARISONS = [("Tnat", "Hnat", "same clean queries: teacher vs labels"),
               ("T", "Hnat", "teacher on all 5000 vs labels on the clean 2593"),
               ("T", "H", "teacher vs labels with injected positives"),
               ("Hnat", "H", "labels: clean 2593 vs 5000 with injection"),
               ("HT", "T", "does a label epoch before the teacher help")]
for a, b, label in COMPARISONS:
    rows, ps = [], []
    for tag in TAGS:
        A, B = S.get(f"{tag}-{a}"), S.get(f"{tag}-{b}")
        if A and B:
            p = ttest_rel(A["vec"], B["vec"]).pvalue
            rows.append((tag, float(np.mean(A["vec"] - B["vec"])), p))
            ps.append(p)
    if not rows:
        continue
    print(f"\n{a} - {b}  ({label})")
    for (tag, diff, p), padj in zip(rows, holm(np.array(ps))):
        print(f"  {tag:6s} diff={diff:+.3f}  p={p:.4f}  p_Holm={padj:.4f}")
        summary[f"test_{a}_vs_{b}_{tag}"] = {"diff": diff, "p": float(p), "p_holm": float(padj)}

# every T student against the teacher
rows, ps = [], []
for tag in TAGS:
    A = S.get(f"{tag}-T")
    if A:
        p = ttest_rel(A["vec"], S["Teacher"]["vec"]).pvalue
        rows.append((tag, float(np.mean(A["vec"] - S["Teacher"]["vec"])), p))
        ps.append(p)
if rows:
    print("\nT - Teacher  (gap to the teacher)")
    for (tag, diff, p), padj in zip(rows, holm(np.array(ps))):
        print(f"  {tag:6s} diff={diff:+.3f}  p={p:.4f}  p_Holm={padj:.4f}")
        summary[f"test_T_vs_Teacher_{tag}"] = {"diff": diff, "p": float(p), "p_holm": float(padj)}

with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# figure
os.makedirs("results/figures", exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5})   # fits the LNCS text width (~12 cm)
fig, ax = plt.subplots(figsize=(4.8, 3.9))
LABELS = {"T": "Trained on the teacher, 5,000 queries", "Tnat": "Trained on the teacher, 2,593 clean queries",
          "Hnat": "Trained on human labels, 2,593 clean queries", "H": "Trained on human labels, 5,000 queries (control)"}
for cond, style in [("T", "o-"), ("Tnat", "s--"), ("Hnat", "^--"), ("H", "x:")]:
    tags = [t for t in TAGS if f"{t}-{cond}" in S]
    if not tags:
        continue
    xs, ys, es = [], [], []
    for t in tags:
        per_seed = [(split_mean(pq, "dl19") + split_mean(pq, "dl20")) / 2 for pq, _, _ in S[f"{t}-{cond}"]["seeds"]]
        xs.append(PARAMS[t]); ys.append(np.mean(per_seed)); es.append(np.std(per_seed))
    ax.errorbar(xs, ys, yerr=es, fmt=style, capsize=3, label=LABELS[cond])
ax.axhline((t19 + t20) / 2, color="k", lw=1, label="Teacher: GPT-5.4 mini")
b = summary["BM25"]
ax.axhline((b["dl19"] + b["dl20"]) / 2, color="gray", ls=":", lw=1, label="BM25, no reranking")
if "Ref-MiniLM-L6 (500k labels)" in summary:
    r = summary["Ref-MiniLM-L6 (500k labels)"]
    ax.scatter([22.7], [(r["dl19"] + r["dl20"]) / 2], marker="*", s=130, color="C4",
               label="Reference: MiniLM-L6, ~500k labeled queries", zorder=5)
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.015)                       # room above the reference star
ax.set_xscale("log")
present = [t for t in TAGS if any(f"{t}-{c}" in S for c in ("T", "H"))]
ax.set_xticks([PARAMS[t] for t in present])
ax.set_xticklabels([f"{PARAMS[t]:g}M\nBERT-{t}" for t in present])
ax.minorticks_off()
ax.set_xlabel("Student parameters")
ax.set_ylabel("nDCG@10 (mean of 2019 and 2020 tests)")
ax.grid(alpha=0.25, lw=0.5)
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig("results/figures/size_quality.png", dpi=300)
fig.savefig("results/figures/size_quality.pdf")
print("\ntable: results/summary.json   figure: results/figures/size_quality.png")
