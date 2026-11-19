"""Step 1: BM25 index over the whole MS MARCO collection (8.8M passages). Run once, 20-40 min."""
import os, sys
sys.path.insert(0, os.getcwd())
from src.retrieval import build_index

print("passages indexed:", build_index("indexes/bm25_msmarco"))
