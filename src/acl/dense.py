"""Dense retrieval with a frozen code-embedding model (CodeRankEmbed).

Most files are identical across the snapshots of a repository, so chunks are
embedded once per distinct document text (sha1 of the text), per repository,
and the cache is reused by every task of that repository. A chunk's document is
its path followed by its source, mirroring the BM25 document (the qualified
name is already in the source).

Writes
  data/dense/<model>/emb/<repo>.npz      unique document embeddings (fp16) and their hashes
  data/dense/<model>/scores/<id>.npy     cosine score per chunk, in chunk_repo order
                                         (the row order of chunks.parquet)
  data/dense/<model>/scores/<id>.json    chunk count and a hash of the chunk ids, to check alignment

Runs on a GPU node; scores are then copied back next to data/prepared.

Usage: python -m acl.dense [--repo owner/name ...] [--batch-size 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from acl import chunking, snapshot

warnings.filterwarnings("ignore", category=SyntaxWarning)

MODEL = "nomic-ai/CodeRankEmbed"
MODEL_TAG = "coderankembed"
# From the model card: queries carry this prefix, code documents carry none.
QUERY_PREFIX = "Represent this query for searching relevant code: "
DOC_MAX_TOKENS = 512
QUERY_MAX_TOKENS = 1024
OUT = Path("data/dense") / MODEL_TAG


def doc_text(c: chunking.Chunk) -> str:
    return f"{c.path}\n{c.text}"


def doc_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def ids_hash(ids: list[str]) -> str:
    return hashlib.sha1("\n".join(ids).encode()).hexdigest()


def load_scores(instance_id: str, chunk_ids: list[str], root: Path = OUT) -> np.ndarray:
    """Dense scores aligned with `chunk_ids` (the id column of chunks.parquet)."""
    meta = json.loads((root / "scores" / f"{instance_id}.json").read_text())
    assert meta["ids_hash"] == ids_hash(chunk_ids), f"dense scores out of sync for {instance_id}"
    return np.load(root / "scores" / f"{instance_id}.npy")


def run_repo(model, repo: str, tasks: list[dict], batch_size: int, out: Path = OUT) -> None:
    emb_path = out / "emb" / f"{repo.replace('/', '__')}.npz"
    score_dir = out / "scores"
    todo = [t for t in tasks if not (score_dir / f"{t['instance_id']}.json").exists()]
    if not todo:
        return
    mirror = snapshot.mirror_path(snapshot.MIRRORS, repo)

    # Pass 1: chunk every snapshot and collect the distinct documents.
    t0 = time.time()
    # Per task only the id checksum and the index of each chunk's document are
    # kept: django has 207 snapshots of ~33k chunks each.
    per_task: dict[str, tuple[str, np.ndarray]] = {}
    texts: dict[str, str] = {}
    index: dict[str, int] = {}
    for t in todo:
        chunks, _ = chunking.chunk_repo(snapshot.read_python_files(mirror, t["base_commit"]))
        rows = np.empty(len(chunks), dtype=np.int32)
        for i, c in enumerate(chunks):
            d = doc_text(c)
            h = doc_hash(d)
            if h not in index:
                index[h] = len(index)
                texts[h] = d
            rows[i] = index[h]
        per_task[t["instance_id"]] = (ids_hash([c.id for c in chunks]), rows)
    hash_of = list(index)

    cache: dict[str, int] = {}
    vecs = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float16)
    if emb_path.exists():
        z = np.load(emb_path)
        vecs = z["vecs"]
        cache = {h: i for i, h in enumerate(z["hashes"].tolist())}
    new = [h for h in texts if h not in cache]
    print(f"{repo}: {len(todo)} tasks, {len(texts)} distinct chunks, {len(new)} to embed "
          f"(chunking {time.time() - t0:.0f}s)", flush=True)

    # Pass 2: embed the new documents.
    if new:
        t1 = time.time()
        model.max_seq_length = DOC_MAX_TOKENS
        e = model.encode([texts[h] for h in new], batch_size=batch_size, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        cache.update({h: len(cache) + i for i, h in enumerate(new)})
        vecs = np.concatenate([vecs, e])
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(emb_path, vecs=vecs, hashes=np.array(list(cache)))
        print(f"{repo}: embedded {len(new)} in {time.time() - t1:.0f}s", flush=True)

    # Pass 3: score each task's chunks against its issue.
    model.max_seq_length = QUERY_MAX_TOKENS
    queries = model.encode([QUERY_PREFIX + t["problem_statement"] for t in todo], batch_size=8,
                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    score_dir.mkdir(parents=True, exist_ok=True)
    vecs32 = vecs.astype(np.float32)
    to_vec = np.array([cache[h] for h in hash_of], dtype=np.int64)
    for t, q in zip(todo, queries):
        idh, rows = per_task[t["instance_id"]]
        s = vecs32[to_vec[rows]] @ q.astype(np.float32)
        np.save(score_dir / f"{t['instance_id']}.npy", s.astype(np.float32))
        (score_dir / f"{t['instance_id']}.json").write_text(json.dumps(
            {"n_chunks": len(rows), "ids_hash": idh, "model": MODEL}))


def _restore_attention_mask_helper(model) -> None:
    """CodeRankEmbed's remote modelling code calls `get_extended_attention_mask`,
    which transformers 5 removed. Restore the transformers 4 behaviour for an
    encoder: a (batch, 1, 1, seq) additive mask, 0 for real tokens and the
    dtype's minimum for padding."""
    import torch
    module = model[0]
    cls = type(getattr(module, "auto_model", None) or module.model)
    if hasattr(cls, "get_extended_attention_mask"):
        return

    def get_extended_attention_mask(self, attention_mask, input_shape=None, *args, **kwargs):
        m = attention_mask[:, None, None, :].to(self.dtype)
        return (1.0 - m) * torch.finfo(self.dtype).min
    cls.get_extended_attention_mask = get_extended_attention_mask


def load_model(device: str):
    from sentence_transformers import SentenceTransformer  # heavy import, GPU node only
    model = SentenceTransformer(MODEL, trust_remote_code=True, device=device)
    _restore_attention_mask_helper(model)
    if device.startswith("cuda"):
        model.half()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks.jsonl")
    ap.add_argument("--repo", action="append")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = load_model(args.device)

    by_repo: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.tasks):
        t = json.loads(line)
        by_repo[t["repo"]].append(t)
    for repo in args.repo or sorted(by_repo, key=lambda r: len(by_repo[r])):
        run_repo(model, repo, by_repo[repo], args.batch_size)


if __name__ == "__main__":
    main()
