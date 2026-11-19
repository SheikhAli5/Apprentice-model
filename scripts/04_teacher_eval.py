"""Step 4: the teacher on the test sets. Reranks the same BM25 top-100 with a sliding window
(30 passages, step 15, 6 calls per query) -> results/runs/teacher_{split}.json.
Answers are cached in cache/teacher/{model}_{split}_top100.jsonl."""
import json, os, sys
sys.path.insert(0, os.getcwd())
import ir_datasets, ir_measures
from ir_measures import nDCG, RR
from dotenv import load_dotenv
from src.data import load_query_texts
from src.teacher import LLMReranker, sliding_rerank

load_dotenv()
MODEL = "gpt-5.4-mini"
docs = ir_datasets.load("msmarco-passage").docs_store()
SPLITS = {"dl19": ("msmarco-passage/trec-dl-2019/judged", nDCG @ 10),
          "dl20": ("msmarco-passage/trec-dl-2020/judged", nDCG @ 10),
          "val": ("msmarco-passage/train/split200-valid", RR @ 10)}
for split, (name, metric) in SPLITS.items():
    ds = ir_datasets.load(name)
    bm25 = json.load(open(f"results/runs/bm25_{split}.json"))
    qtext = load_query_texts(name, list(bm25))
    teacher = LLMReranker(MODEL, f"cache/teacher/{MODEL}_{split}_top100.jsonl")
    run, calls, per_query_sec = {}, [], []
    for qid, cands in bm25.items():
        ranking, recs = sliding_rerank(teacher, qid, qtext[qid], sorted(cands, key=cands.get, reverse=True), docs)
        calls += recs
        per_query_sec.append(sum(r["sec"] for r in recs))
        run[qid] = {d: len(ranking) - r for r, d in enumerate(ranking)}
    json.dump(run, open(f"results/runs/teacher_{split}.json", "w"))
    score = ir_measures.calc_aggregate([metric], list(ds.qrels_iter()), run)[metric]
    cost = sum(r["in_tok"] for r in calls) / 1e6 * 0.75 + sum(r["out_tok"] for r in calls) / 1e6 * 4.5
    print(f"{split}: {metric}={score:.3f}  ${cost:.2f}  {sum(per_query_sec) / len(per_query_sec):.1f}s/query  "
          f"incomplete windows: {sum(r['n_missing'] > 0 for r in calls)} of {len(calls)}")
