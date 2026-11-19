"""Step 2: first-stage candidates.
dl19, dl20, val: BM25 top-100 -> results/runs/bm25_{split}.json, every reranker uses these.
train: 5000 random queries, BM25 top-30. If the labeled passage is not there it replaces
the 30th one -> data/train_candidates.jsonl ("injected" field)."""
import json, os, random, sys
sys.path.insert(0, os.getcwd())
import ir_datasets, ir_measures
from ir_measures import nDCG, RR, R
from src.data import load_positives, load_query_texts, ensure_positive, save_jsonl
from src.retrieval import BM25Retriever

SEED, N_TRAIN, K_TRAIN, K_EVAL = 42, 5000, 30, 100
TRAIN_DS = "msmarco-passage/train/split200-train"
EVAL = {"dl19": "msmarco-passage/trec-dl-2019/judged",
        "dl20": "msmarco-passage/trec-dl-2020/judged",
        "val": "msmarco-passage/train/split200-valid"}
os.makedirs("results/runs", exist_ok=True)
os.makedirs("data", exist_ok=True)
bm25 = BM25Retriever("indexes/bm25_msmarco")

for split, name in EVAL.items():
    ds = ir_datasets.load(name)
    qids = sorted(load_positives(name)) if split == "val" else [q.query_id for q in ds.queries_iter()]
    qtext = load_query_texts(name, qids)
    res = bm25.retrieve([qtext[q] for q in qids], k=K_EVAL)
    run = {q: dict(r) for q, r in zip(qids, res)}
    with open(f"results/runs/bm25_{split}.json", "w") as f:
        json.dump(run, f)
    metrics = [RR @ 10] if split == "val" else [nDCG @ 10, R(rel=2) @ 100]
    print(split, len(qids), ir_measures.calc_aggregate(metrics, list(ds.qrels_iter()), run))

train_pos = load_positives(TRAIN_DS)
train_qids = random.Random(SEED).sample(sorted(train_pos), N_TRAIN)
train_text = load_query_texts(TRAIN_DS, train_qids)
results = bm25.retrieve([train_text[q] for q in train_qids], k=K_TRAIN)
rows, n_inj = [], 0
for qid, res in zip(train_qids, results):
    cands, injected = ensure_positive([d for d, _ in res], train_pos[qid])
    n_inj += injected
    rows.append({"qid": qid, "query": train_text[qid], "candidates": cands,
                 "positives": train_pos[qid], "injected": injected})
save_jsonl(rows, "data/train_candidates.jsonl")
print(f"train: {len(rows)} queries, positive injected in {n_inj / len(rows):.1%}")
