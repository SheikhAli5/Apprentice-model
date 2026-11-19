"""Step 3: pilot to choose the teacher. 3 models x 124 val queries x shuffled BM25 top-30.
Answers are cached in cache/teacher/{model}_val.jsonl, so running it again costs nothing."""
import json, os, random, sys
sys.path.insert(0, os.getcwd())
import ir_datasets, ir_measures
from ir_measures import RR
from dotenv import load_dotenv
from src.data import load_query_texts
from src.teacher import LLMReranker

load_dotenv()
SEED, K = 42, 30
VAL_DS = "msmarco-passage/train/split200-valid"
PRICES = {"gpt-5.4-nano": (0.20, 1.25), "gpt-5.4-mini": (0.75, 4.50), "gpt-5.4": (2.50, 15.00)}  # $ per 1M tokens (in, out)
os.makedirs("cache/teacher", exist_ok=True)
docs = ir_datasets.load("msmarco-passage").docs_store()
val_qrels = list(ir_datasets.load(VAL_DS).qrels_iter())
bm25_val = json.load(open("results/runs/bm25_val.json"))
val_text = load_query_texts(VAL_DS, list(bm25_val))

for model in PRICES:
    teacher = LLMReranker(model, f"cache/teacher/{model}_val.jsonl")
    run, recs = {}, []
    for qid, cands in bm25_val.items():
        top = sorted(cands, key=cands.get, reverse=True)[:K]
        random.Random(f"{SEED}-{qid}").shuffle(top)
        rec = teacher.rerank(qid, val_text[qid], [docs.get(d).text for d in top])
        recs.append(rec)
        run[qid] = {top[i]: K - r for r, i in enumerate(rec["order"])}
    rr = ir_measures.calc_aggregate([RR @ 10], val_qrels, run)[RR @ 10]
    tin, tout = sum(r["in_tok"] for r in recs), sum(r["out_tok"] for r in recs)
    cost = tin / 1e6 * PRICES[model][0] + tout / 1e6 * PRICES[model][1]
    print(f"{model:13s} RR@10={rr:.3f}  ${cost:.2f}  {sum(r['sec'] for r in recs) / len(recs):.1f}s/query  "
          f"incomplete: {sum(r['n_missing'] > 0 for r in recs)}")
