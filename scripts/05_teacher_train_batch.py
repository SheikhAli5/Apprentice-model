"""Step 5: teacher rankings for the 5000 training queries with the OpenAI Batch API (50% cheaper).
  python scripts/05_teacher_train_batch.py prepare   # request files of 1000 + the shuffled orders
  python scripts/05_teacher_train_batch.py submit    # only once, submitting again means paying again
  python scripts/05_teacher_train_batch.py status
  python scripts/05_teacher_train_batch.py collect   # -> cache/teacher/gpt-5.4-mini_train.jsonl"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from dotenv import load_dotenv
from src.teacher import build_batch_requests, collect_batch

MODEL, CHUNK, B = "gpt-5.4-mini", 1000, "data/batch"
CACHE = f"cache/teacher/{MODEL}_train.jsonl"
cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
load_dotenv()

if cmd == "prepare":
    import ir_datasets
    os.makedirs(B, exist_ok=True)
    rows = [json.loads(l) for l in open("data/train_candidates.jsonl")]
    shown, requests = build_batch_requests(rows, ir_datasets.load("msmarco-passage").docs_store(), MODEL)
    json.dump(shown, open(f"{B}/train_shown_order.json", "w"))
    for k in range(0, len(requests), CHUNK):
        with open(f"{B}/train_part{k // CHUNK}.jsonl", "w") as f:
            for r in requests[k:k + CHUNK]:
                f.write(json.dumps(r) + "\n")
    print(len(requests), "requests written")
    sys.exit()

import openai
client = openai.OpenAI()
if cmd == "submit":
    if os.path.exists(f"{B}/batch_ids.json"):
        sys.exit("batch_ids.json exists, the batches were already submitted. Delete it by hand only if you are sure.")
    ids = []
    for k in range(5):
        file = client.files.create(file=open(f"{B}/train_part{k}.jsonl", "rb"), purpose="batch")
        ids.append(client.batches.create(input_file_id=file.id, endpoint="/v1/responses",
                                         completion_window="24h").id)
    json.dump(ids, open(f"{B}/batch_ids.json", "w"))
    print(ids)
elif cmd == "status":
    for bid in json.load(open(f"{B}/batch_ids.json")):
        b = client.batches.retrieve(bid)
        print(bid, b.status, b.request_counts)
elif cmd == "collect":
    shown = json.load(open(f"{B}/train_shown_order.json"))
    done = collect_batch(client, json.load(open(f"{B}/batch_ids.json")), shown, CACHE)
    recs = list(done.values())
    cost = sum(r["in_tok"] for r in recs) / 1e6 * 0.375 + sum(r["out_tok"] for r in recs) / 1e6 * 2.25
    print(f"done {len(done)} of {len(shown)} | incomplete {sum(r['n_missing'] > 0 for r in recs)} | ${cost:.2f}")
