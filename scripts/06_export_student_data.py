"""Step 6: self-contained files for the students, with the passage texts inside,
so training doesn't need the index or ir_datasets anymore.
  data/train_student.json        query, 30 passages, labeled positive, teacher order
  data/eval_{val,dl19,dl20}.json same BM25 top-100 + qrels"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import ir_datasets
from src.data import load_query_texts

docs = ir_datasets.load("msmarco-passage").docs_store()
teacher = {}
for line in open("cache/teacher/gpt-5.4-mini_train.jsonl"):
    r = json.loads(line)
    teacher[r["qid"]] = r["ranked_docs"]
train = []
for line in open("data/train_candidates.jsonl"):
    row = json.loads(line)
    if row["qid"] not in teacher:
        continue
    cands = row["candidates"]
    train.append({"qid": row["qid"], "query": row["query"],
                  "docs": {d: docs.get(d).text for d in cands},
                  "pos": next(d for d in row["positives"] if d in cands),
                  "teacher": teacher[row["qid"]]})
json.dump(train, open("data/train_student.json", "w"))
print("train:", len(train))

SPLITS = {"val": "msmarco-passage/train/split200-valid",
          "dl19": "msmarco-passage/trec-dl-2019/judged",
          "dl20": "msmarco-passage/trec-dl-2020/judged"}
for split, name in SPLITS.items():
    bm25 = json.load(open(f"results/runs/bm25_{split}.json"))
    qtext = load_query_texts(name, list(bm25))
    data = {q: {"query": qtext[q], "docs": {d: docs.get(d).text for d in c}} for q, c in bm25.items()}
    qrels = {}
    for q in ir_datasets.load(name).qrels_iter():
        qrels.setdefault(q.query_id, {})[q.doc_id] = q.relevance
    json.dump({"data": data, "qrels": qrels}, open(f"data/eval_{split}.json", "w"))
    print(split, len(data), "queries")
