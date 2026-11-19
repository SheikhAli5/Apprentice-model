"""First stage: BM25 with bm25s."""
import bm25s
import numpy as np
import Stemmer


class BM25Retriever:
    def __init__(self, index_dir: str):
        self.retriever = bm25s.BM25.load(index_dir, mmap=True)  # read from disk, not loaded into RAM
        self.doc_ids = np.load(f"{index_dir}/doc_ids.npy")      # bm25s returns positions, not ids
        self.stemmer = Stemmer.Stemmer("english")               # same as for indexing

    def retrieve(self, query_texts, k=100):
        """Top-k (doc_id, score) pairs for each query, best first."""
        tokens = bm25s.tokenize(query_texts, stopwords="en", stemmer=self.stemmer)
        positions, scores = self.retriever.retrieve(tokens, k=k)
        return [[(str(self.doc_ids[p]), float(s)) for p, s in zip(pos_row, sc_row)]
                for pos_row, sc_row in zip(positions, scores)]

def build_index(index_dir="indexes/bm25_msmarco", k1=0.9, b=0.4):
    # run once: index over all 8.8M MS MARCO passages
    # k1=0.9, b=0.4 are the usual Anserini/Pyserini values for MS MARCO
    import ir_datasets
    doc_ids, texts = [], []
    for doc in ir_datasets.load("msmarco-passage").docs_iter():
        doc_ids.append(doc.doc_id)
        texts.append(doc.text)
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"))
    del texts
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(tokens)
    retriever.save(index_dir)
    np.save(f"{index_dir}/doc_ids.npy", np.array(doc_ids))
    return len(doc_ids)
