"""Queries, labels and BM25 candidates for training."""
import json
from collections import defaultdict

import ir_datasets


def load_positives(dataset_name):
    """{query_id: [doc_id, ...]} with the passages labeled as relevant."""
    positives = defaultdict(list)
    dataset = ir_datasets.load(dataset_name)
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0:
            positives[qrel.query_id].append(qrel.doc_id)
    return dict(positives)


def load_query_texts(dataset_name, query_ids):
    wanted = set(query_ids)
    texts = {}
    dataset = ir_datasets.load(dataset_name)
    for q in dataset.queries_iter():  # ~800k queries for train, so only keep the wanted ones
        if q.query_id in wanted:
            texts[q.query_id] = q.text
    return texts


def ensure_positive(candidates, positives):
    # if BM25 missed the labeled passage, it takes the place of the last candidate
    if any(p in candidates for p in positives):
        return candidates, False

    new_candidates = list(candidates)
    new_candidates[-1] = positives[0]
    return new_candidates, True


def save_jsonl(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
