import json
import os
import re
import time
import random

import openai


def build_prompt(query, passages):
    n = len(passages)
    lines = []
    for i, text in enumerate(passages, start=1):
        lines.append(f"[{i}] {text}")
    passages_block = "\n".join(lines)
    return (
        f"Below are {n} passages, each marked with an identifier in brackets.\n"
        f"Search query: {query}\n\n"
        f"{passages_block}\n\n"
        f"Search query: {query}\n"
        f"Rank ALL {n} passages from most to least relevant to the query. "
        "Answer only with identifiers in the format [3] > [1] > [2], with no explanation."
    )


def parse(text, n):
    numbers = re.findall(r"\[(\d+)\]", text)
    seen = set()
    order = []
    for num in numbers:
        i = int(num) - 1
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    n_missing = n - len(seen)
    for i in range(n):
        if i not in seen:
            order.append(i)
    return order, n_missing


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                cache[rec["qid"]] = rec
    return cache


def append_cache(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


class LLMReranker:
    def __init__(self, model, cache_path):
        self.model = model
        self.cache_path = cache_path
        self.cache = load_cache(cache_path)
        self.client = openai.OpenAI()

    def call(self, prompt):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": "low"},
            max_output_tokens=4000,
        )
        u = response.usage
        return response.output_text, u.input_tokens, u.output_tokens

    def rerank(self, qid, query, passages):
        if qid in self.cache:                             # already paid for
            return self.cache[qid]
        prompt = build_prompt(query, passages)
        t0 = time.time()
        text, in_tok, out_tok = self.call(prompt)
        sec = time.time() - t0
        order, n_missing = parse(text, len(passages))
        rec = {"qid": qid, "order": order, "n_missing": n_missing, "raw": text,
               "in_tok": in_tok, "out_tok": out_tok, "sec": round(sec, 2)}
        self.cache[qid] = rec
        append_cache(self.cache_path, rec)
        return rec

def sliding_rerank(teacher, qid, query, doc_ids, docs, window=30, step=15, seed=42):
    """Reranks a long list with overlapping windows, from the bottom up like RankGPT.
    Returns the doc ids best to worst and the records of all calls."""
    ranking = list(doc_ids)                         # BM25 order
    start = max(len(ranking) - window, 0)           # bottom window first (70..99)
    recs = []
    while True:
        end = start + window
        shown = ranking[start:end]
        random.Random(f"{seed}-{qid}-{start}").shuffle(shown)       # hide the BM25 order from the teacher
        rec = teacher.rerank(f"{qid}|{start}", query,               # cache key = query + window
                             [docs.get(d).text for d in shown])
        recs.append(rec)
        ranking[start:end] = [shown[i] for i in rec["order"]]
        if start == 0:
            break
        start = max(start - step, 0)
    return ranking, recs

# Batch API for the 5000 training queries (half the price)

def build_batch_requests(rows, docs, model, seed=42):
    """rows from train_candidates.jsonl. Returns (shown, requests):
    shown[qid] = doc ids in the shuffled order the teacher sees, needed to decode the answer,
    requests = JSONL lines for the Batch API, same parameters as LLMReranker.call."""
    shown, requests = {}, []
    for row in rows:
        qid = row["qid"]
        cands = list(row["candidates"])
        random.Random(f"{seed}-{qid}").shuffle(cands)      # same shuffle as in the pilot
        shown[qid] = cands
        prompt = build_prompt(row["query"], [docs.get(d).text for d in cands])
        requests.append({"custom_id": qid, "method": "POST", "url": "/v1/responses",
                         "body": {"model": model, "input": prompt,
                                  "reasoning": {"effort": "low"}, "max_output_tokens": 4000}})
    return shown, requests


def batch_output_text(body):
    # raw batch output has no .output_text, take the message text and skip the reasoning items
    for item in body["output"]:
        if item["type"] == "message":
            for c in item["content"]:
                if c["type"] == "output_text":
                    return c["text"]
    return ""


def collect_batch(client, batch_ids, shown, cache_path):
    """Downloads finished batches, parses the answers and appends them to the cache."""
    done = load_cache(cache_path)
    for bid in batch_ids:
        b = client.batches.retrieve(bid)
        if b.status != "completed":
            print(bid, "not ready yet:", b.status)
            continue
        for line in client.files.content(b.output_file_id).text.splitlines():
            res = json.loads(line)
            qid = res["custom_id"]
            if qid in done or res.get("error") or res["response"]["status_code"] != 200:
                continue
            body = res["response"]["body"]
            text = batch_output_text(body)
            order, n_missing = parse(text, len(shown[qid]))
            rec = {"qid": qid, "order": order, "ranked_docs": [shown[qid][i] for i in order],
                   "n_missing": n_missing, "raw": text,
                   "in_tok": body["usage"]["input_tokens"], "out_tok": body["usage"]["output_tokens"]}
            append_cache(cache_path, rec)
            done[qid] = rec
    return done
