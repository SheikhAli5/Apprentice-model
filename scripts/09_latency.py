"""Step 9: student latency on the Mac, ms per query (100 passages), CPU and MPS.
Speed depends on the architecture and not on the weights, so the pretrained checkpoints are enough.
Run from the repo root: python scripts/09_latency.py"""
import json, os, sys, time
import torch
sys.path.insert(0, os.getcwd())
from src.student import CrossEncoder, rerank

MODELS = {"tiny": "google/bert_uncased_L-2_H-128_A-2",
          "mini": "google/bert_uncased_L-4_H-256_A-4",
          "small": "google/bert_uncased_L-4_H-512_A-8",
          "base": "bert-base-uncased",
          "ref_MiniLM-L6": "cross-encoder/ms-marco-MiniLM-L6-v2"}
ev = json.load(open("data/eval_dl19.json"))["data"]
qids = list(ev)[:20]                                    # 20 queries x 100 passages
sub = {q: ev[q] for q in qids}
devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
out = {}
for tag, name in MODELS.items():
    ce = CrossEncoder(name)
    params = sum(p.numel() for p in ce.model.parameters()) / 1e6
    out[tag] = {"params_M": round(params, 2)}
    for device in devices:
        ce.device = device
        ce.model.to(device)
        rerank(ce, {qids[0]: ev[qids[0]]})              # warm-up
        t0 = time.time()
        rerank(ce, sub)
        ms = (time.time() - t0) / len(sub) * 1000
        out[tag][f"ms_per_query_{device}"] = round(ms, 1)
        print(f"{tag:14s} {params:7.1f}M  {device}: {ms:8.1f} ms per query", flush=True)
    del ce
out["teacher_gpt-5.4-mini"] = {"ms_per_query_api": 15700, "usd_per_query": 0.019,
                               "note": "6 windows of 30, from the cached DL19/DL20 runs"}
with open("results/latency.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved: results/latency.json")
