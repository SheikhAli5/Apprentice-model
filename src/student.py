"""Student cross-encoder: losses, training stages, reranking."""
import copy, json, os, random, time

import torch
import torch.nn.functional as F
import ir_measures
from ir_measures import nDCG, RR
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoder:
    def __init__(self, name, max_len=192):
        self.device = get_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=1).to(self.device)          # one output = relevance score
        self.max_len = max_len

    def scores(self, query, passages):
        # [CLS] query [SEP] passage [SEP] for every passage, returns shape (n,)
        enc = self.tok([query] * len(passages), passages, truncation="only_second",
                       max_length=self.max_len, padding=True, return_tensors="pt").to(self.device)
        return self.model(**enc).logits.squeeze(-1)


# losses

def infonce_loss(scores, pos_idx):
    # human labels: softmax over the 30 candidates, the labeled passage is the target
    target = torch.tensor([pos_idx], device=scores.device)
    return F.cross_entropy(scores.unsqueeze(0), target)


def ranknet_loss(scores, teacher_order):
    # teacher_order = candidate indices from best to worst according to the teacher.
    # each pair (a above b) costs log(1 + exp(s_b - s_a)), mean over the 435 pairs
    s = scores[teacher_order]
    diff = s.unsqueeze(1) - s.unsqueeze(0)         # diff[a, b] = s_a - s_b
    above = torch.triu(torch.ones_like(diff, dtype=torch.bool), diagonal=1)
    return F.softplus(-diff[above]).mean()


# training and reranking

def train_stage(ce, train_data, mode, lr, seed, q_per_step=4, warmup=0.1):
    """One epoch. mode 'H' = human labels, 'T' = teacher order."""
    order = list(range(len(train_data)))
    random.Random(seed).shuffle(order)
    steps = (len(order) + q_per_step - 1) // q_per_step
    opt = torch.optim.AdamW(ce.model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    use_amp = ce.device == "cuda"                          # fp16 only on the Colab GPU
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ce.model.train()
    t0, running = time.time(), 0.0
    for step in range(steps):
        batch = [train_data[i] for i in order[step * q_per_step:(step + 1) * q_per_step]]
        with torch.autocast(device_type="cuda" if use_amp else "cpu",
                            dtype=torch.float16, enabled=use_amp):
            loss = 0.0
            for ex in batch:
                ids = list(ex["docs"])
                s = ce.scores(ex["query"], [ex["docs"][d] for d in ids]).float()
                if mode == "H":
                    loss = loss + infonce_loss(s, ids.index(ex["pos"]))
                else:
                    where = {d: i for i, d in enumerate(ids)}
                    loss = loss + ranknet_loss(s, [where[d] for d in ex["teacher"]])
            loss = loss / len(batch)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(ce.model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        if (step + 1) % 250 == 0:
            print(f"    {mode} step {step + 1}/{steps}  loss {running / 250:.4f}  {time.time() - t0:.0f}s")
            running = 0.0


@torch.no_grad()
def rerank(ce, eval_data, batch=128):
    """Scores all candidates of every query, returns a run {qid: {doc_id: score}}."""
    ce.model.eval()
    use_amp = ce.device == "cuda"
    run = {}
    for qid, item in eval_data.items():
        ids = list(item["docs"])
        out = []
        for k in range(0, len(ids), batch):
            chunk = ids[k:k + batch]
            with torch.autocast(device_type="cuda" if use_amp else "cpu",
                                dtype=torch.float16, enabled=use_amp):
                out.append(ce.scores(item["query"], [item["docs"][d] for d in chunk]).float().cpu())
        run[qid] = dict(zip(ids, torch.cat(out).tolist()))
    return run


def evaluate(run, qrels, split):
    # val has a single label per query, so RR@10 there
    metric = RR @ 10 if split == "val" else nDCG @ 10
    return ir_measures.calc_aggregate([metric], qrels, run)[metric]


def _snapshot(ce):
    return {k: v.detach().cpu().clone() for k, v in ce.model.state_dict().items()}


def run_model(name, tag, lr, seed, train_data, evals, out_dir="results/runs"):
    """One model + seed, stages H1, H, HT, T1, T. Evaluates and saves runs after each one."""
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(f"{out_dir}/student_{tag}_T_s{seed}_dl20.json"):
        print(f"{tag} s{seed}: already done, skipping")
        return

    def finish(cond):
        for split, ev in evals.items():
            run = rerank(ce, ev["data"])
            with open(f"{out_dir}/student_{tag}_{cond}_s{seed}_{split}.json", "w") as f:
                json.dump(run, f)
            print(f"  {tag} {cond:3s} s{seed} {split}: {evaluate(run, ev['qrels'], split):.3f}")

    torch.manual_seed(seed)                     # before creating the model, the new head depends on it
    ce = CrossEncoder(name)
    init = _snapshot(ce)
    print(f"=== {tag} seed {seed} on {ce.device} ===")

    train_stage(ce, train_data, "H", lr, seed * 10 + 1)
    finish("H1")
    h1 = _snapshot(ce)
    train_stage(ce, train_data, "H", lr, seed * 10 + 2)
    finish("H")                                 # 2 epochs on labels

    ce.model.load_state_dict(h1)
    train_stage(ce, train_data, "T", lr, seed * 10 + 3)
    finish("HT")                                # 1 epoch labels + 1 epoch teacher

    ce.model.load_state_dict(init)
    train_stage(ce, train_data, "T", lr, seed * 10 + 4)
    finish("T1")
    train_stage(ce, train_data, "T", lr, seed * 10 + 5)
    finish("T")                                 # 2 epochs on the teacher

# "natural" subset (= clean queries in the report): BM25 found the positive by itself

def natural_subset(train_data, candidates_path="data/train_candidates.jsonl"):
    natural = set()
    with open(candidates_path) as f:
        for line in f:
            r = json.loads(line)
            if not r["injected"]:
                natural.add(r["qid"])
    return [ex for ex in train_data if ex["qid"] in natural]


def run_nat(name, tag, lr, seed, train_nat, evals, out_dir="results/runs"):
    """H-nat and T-nat: 2 epochs each on the natural subset, same initial head.
    Saves diag_{tag}_{H|T}nat_s{seed}_{split}.json, skips what is already there."""
    os.makedirs(out_dir, exist_ok=True)
    for mode in ["H", "T"]:
        prefix = f"{out_dir}/diag_{tag}_{mode}nat_s{seed}"
        if os.path.exists(prefix + "_dl20.json"):
            print(f"{tag} {mode}-nat s{seed}: already done, skipping")
            continue
        torch.manual_seed(seed)
        ce = CrossEncoder(name)
        print(f"=== {tag} {mode}-nat seed {seed} on {ce.device} ===")
        train_stage(ce, train_nat, mode, lr, seed * 10 + 1)
        train_stage(ce, train_nat, mode, lr, seed * 10 + 2)
        for split, ev in evals.items():
            run = rerank(ce, ev["data"])
            with open(f"{prefix}_{split}.json", "w") as f:
                json.dump(run, f)
            print(f"  {tag} {mode}-nat s{seed} {split}: {evaluate(run, ev['qrels'], split):.3f}")
        del ce
