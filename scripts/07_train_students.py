"""Step 7: trains all the students and scores the reference model. Run files go to results/runs.
Finished runs are skipped, so after a crash just start it again. About 12 h on a MacBook M3 (MPS).
  reference:   cross-encoder/ms-marco-MiniLM-L6-v2, trained on ~500k labeled MS MARCO queries
  main runs:   3 sizes x 3 seeds x {H1, H, HT, T1, T}   (run_model)
  clean data:  3 sizes x 3 seeds x {H-nat, T-nat}       (run_nat, 2593 queries without injection)
BERT-base (110M) does not fit in 16 GB on MPS, see colab/run_base.ipynb."""
import json, os, sys, time
sys.path.insert(0, os.getcwd())
from src.student import CrossEncoder, rerank, evaluate, run_model, natural_subset, run_nat

R = "results/runs"
os.makedirs(R, exist_ok=True)
train = json.load(open("data/train_student.json"))
train_nat = natural_subset(train)
evals = {s: json.load(open(f"data/eval_{s}.json")) for s in ["val", "dl19", "dl20"]}
print("train:", len(train), "| natural:", len(train_nat), flush=True)

MODELS = [("google/bert_uncased_L-2_H-128_A-2", "tiny", 5e-5),     # 4.4M
          ("google/bert_uncased_L-4_H-256_A-4", "mini", 5e-5),     # 11.2M
          ("google/bert_uncased_L-4_H-512_A-8", "small", 5e-5)]    # 28.8M
SEEDS = (1, 2, 3)
t0 = time.time()

# reference model, rerank only
REF, REF_TAG = "cross-encoder/ms-marco-MiniLM-L6-v2", "ref_msmarco-MiniLM-L6"
if not os.path.exists(f"{R}/{REF_TAG}_dl20.json"):
    ce = CrossEncoder(REF)
    for split, ev in evals.items():
        run = rerank(ce, ev["data"])
        with open(f"{R}/{REF_TAG}_{split}.json", "w") as f:
            json.dump(run, f)
        print(f"  reference {split}: {evaluate(run, ev['qrels'], split):.3f}", flush=True)
    del ce

for seed in SEEDS:
    for name, tag, lr in MODELS:
        run_model(name, tag, lr, seed, train, evals)       # H1, H, HT, T1, T
        run_nat(name, tag, lr, seed, train_nat, evals)     # H-nat, T-nat
    print(f"--- seed {seed} done, {(time.time() - t0) / 60:.0f} min", flush=True)
