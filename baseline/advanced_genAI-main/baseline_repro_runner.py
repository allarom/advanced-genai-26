#!/usr/bin/env python3
"""Reproducible baseline runner for multilingual RAG.

Runs and evaluates five baseline methods:
1) BM25
2) Dense Retrieval
3) GraphRAG
4) Hybrid Retrieval (weighted RRF fusion)
5) Re-ranking (Hybrid + overlap rerank)

This script is intentionally minimal and follows the existing project assets
(created in Step-1/Step-2 notebooks) instead of redesigning retrieval logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
import statistics
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import nltk
    from langdetect import detect
    from rank_bm25 import BM25Okapi
except Exception:
    nltk = None
    detect = None
    BM25Okapi = None


@dataclass
class Config:
    root: Path
    bm25_pickle: Path
    dense_loader: Path
    graphrag_loader: Path
    qa_path: Path
    qrels_dir: Path
    output_dir: Path
    seed: int = 42
    k_values: tuple[int, ...] = (1, 3, 5, 10)
    top_n: int = 100
    query_field: str = "question"
    cached_rerank_dir: Path | None = None


# Compatibility layer for notebook-pickled BM25 objects.
class BilingualBM25:
    """Legacy BM25 class used when loading pickles produced in notebooks."""

    def _rank_lang(self, q: str, lang: str, k: int):
        try:
            q_tokens = nltk.word_tokenize(q)
        except Exception:
            q_tokens = q.split()
        scores = self.bm25[lang].get_scores(q_tokens)
        idx = np.argsort(scores)[::-1][:k]
        hits = []
        for i in idx:
            d = self.docs_by_lang[lang][i]
            d.metadata["bm25_score"] = float(scores[i])
            hits.append(d)
        return hits

    def search(self, query: str, top_k: int = 100):
        if detect is None or nltk is None:
            raise RuntimeError("BM25 pickle requires 'langdetect' and 'nltk' installed.")
        src = detect(query) if query.strip() else "en"
        src = src if src in ("en", "de") else "en"
        bag = []
        translator = getattr(self, "translator", None)
        for lang in ("en", "de"):
            q_lang = translator.translate(query, lang) if translator and lang != src else query
            bag.extend(self._rank_lang(q_lang, lang, top_k))

        best = {}
        for d in bag:
            uid = d.metadata.get("chunk_id") or d.metadata.get("record_id")
            if uid not in best or d.metadata["bm25_score"] > best[uid].metadata["bm25_score"]:
                best[uid] = d
        return sorted(best.values(), key=lambda d: d.metadata["bm25_score"], reverse=True)[:top_k]


class QEBM25:
    """Legacy query-expansion BM25 wrapper used in pickled artifacts."""

    @staticmethod
    def _expand_query(query: str, base_retriever, fb_docs: int = 5, fb_terms: int = 5) -> str:
        if nltk is None:
            return query
        try:
            stop_en = set(nltk.corpus.stopwords.words("english"))
            stop_de = set(nltk.corpus.stopwords.words("german"))
        except Exception:
            stop_en, stop_de = set(), set()
        hits = base_retriever.search(query, top_k=fb_docs)
        def _tok(text: str) -> list[str]:
            try:
                return nltk.word_tokenize(text.lower())
            except Exception:
                return text.lower().split()
        tokens = [
            t.lower()
            for h in hits
            for t in _tok(h.page_content)
            if t.isalpha() and t not in stop_en and t not in stop_de
        ]
        extra = " ".join(w for w, _ in nltk.FreqDist(tokens).most_common(fb_terms))
        return f"{query} {extra}" if extra else query

    def search(self, query: str, top_k: int = 100):
        if not hasattr(self, "base"):
            raise AttributeError("Loaded QEBM25 object has no 'base' attribute.")
        expanded = self._expand_query(query, self.base)
        return self.base.search(expanded, top_k)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_fake_langchain_document_modules() -> None:
    """Provide a minimal Document class for unpickling when langchain_core is absent."""
    if "langchain_core.documents.base" in sys.modules:
        return
    pkg = types.ModuleType("langchain_core")
    pkg.__path__ = []
    docs_pkg = types.ModuleType("langchain_core.documents")
    docs_pkg.__path__ = []
    docs_base = types.ModuleType("langchain_core.documents.base")

    class Document:
        def __init__(self, *args, **kwargs):
            pass

    docs_base.Document = Document
    docs_pkg.Document = Document

    sys.modules["langchain_core"] = pkg
    sys.modules["langchain_core.documents"] = docs_pkg
    sys.modules["langchain_core.documents.base"] = docs_base


class GraphAdapter:
    def __init__(self, graph_module: Any):
        self.graph_module = graph_module

    def search(self, query: str, top_k: int = 100):
        if hasattr(self.graph_module, "retrieve"):
            return self.graph_module.retrieve(query, top_k=top_k)
        if hasattr(self.graph_module, "graph_rag") and hasattr(self.graph_module.graph_rag, "retrieve"):
            return self.graph_module.graph_rag.retrieve(query, top_k=top_k)
        raise AttributeError("GraphRAG loader does not expose retrieve(query, top_k)")


class HybridFusionRetriever:
    def __init__(
        self,
        bm25: Any,
        dense: Any,
        graph: Any,
        weights: dict[str, float] | None = None,
        rerank: bool = False,
    ):
        self.bm25 = bm25
        self.dense = dense
        self.graph = graph
        self.weights = weights or {"bm25": 1.2, "dense": 1.0, "graph": 0.6}
        self.rerank = rerank

    @staticmethod
    def _uid(doc: Any) -> str | None:
        if isinstance(doc, tuple):
            doc = doc[0]
        meta = getattr(doc, "metadata", {}) or {}
        return meta.get("chunk_id") or meta.get("record_id") or meta.get("doc_id")

    @staticmethod
    def _text(doc: Any) -> str:
        if isinstance(doc, tuple):
            doc = doc[0]
        meta = getattr(doc, "metadata", {}) or {}
        return (meta.get("original_text") or getattr(doc, "page_content", "") or "").lower()

    def _safe_unique(self, docs: list[Any]) -> list[Any]:
        out = []
        seen = set()
        for d in docs:
            u = self._uid(d)
            if u is None or u in seen:
                continue
            seen.add(u)
            out.append(d)
        return out

    def _rrf_fuse(self, runs: dict[str, list[Any]], k_rrf: int = 60) -> list[Any]:
        scores: dict[str, float] = defaultdict(float)
        store: dict[str, Any] = {}

        for name, docs in runs.items():
            w = float(self.weights.get(name, 1.0))
            for rank, d in enumerate(docs, start=1):
                u = self._uid(d)
                if u is None:
                    continue
                store.setdefault(u, d)
                scores[u] += w * (1.0 / (k_rrf + rank))

        ranked = sorted(store.values(), key=lambda d: scores[self._uid(d)], reverse=True)
        for d in ranked:
            if isinstance(d, tuple):
                d = d[0]
            d.metadata["fused_score"] = float(scores[self._uid(d)])
        return ranked

    def _overlap_rerank(self, docs: list[Any], query: str, top_k: int) -> list[Any]:
        q_terms = {t.lower() for t in query.split() if t.strip()}
        scored = []
        for d in docs:
            d_terms = set(self._text(d).split())
            overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
            scored.append((overlap, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]]

    def search(self, query: str, top_k: int = 100):
        pre_k = max(30, top_k)
        bm25_docs = self._safe_unique(self.bm25.search(query, top_k=pre_k))
        dense_docs = self._safe_unique(self.dense.search(query, top_k=pre_k))
        graph_docs = self._safe_unique(self.graph.search(query, top_k=pre_k))

        fused = self._rrf_fuse({"bm25": bm25_docs, "dense": dense_docs, "graph": graph_docs})

        if self.rerank:
            rerank_k = max(50, top_k)
            fused = self._overlap_rerank(fused[:rerank_k], query, top_k=rerank_k)

        return fused[:top_k]


class MethodAdapter:
    def __init__(self, retriever: Any, method: str = "search"):
        self.retriever = retriever
        self.method = method

    def search(self, query: str, top_k: int = 100):
        fn = getattr(self.retriever, self.method)
        if self.method == "retrieve":
            return fn(query, top_k=top_k)
        return fn(query, top_k=top_k)


def load_qrels(qrels_dir: Path, threshold: float = 0.5) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for fp in sorted(qrels_dir.glob("*.json")):
        doc_id = fp.stem
        payload = json.loads(fp.read_text(encoding="utf-8"))
        for qid, rel_info in payload.items():
            if float(rel_info.get("relevance_score", 0.0)) >= threshold:
                qrels[str(qid)].add(doc_id)
    return qrels


def extract_doc_id(doc: Any) -> str | None:
    if isinstance(doc, tuple):
        doc = doc[0]
    meta = getattr(doc, "metadata", {}) or {}
    return meta.get("chunk_id") or meta.get("record_id") or meta.get("doc_id")


def normalize_docs(results: list[Any], top_n: int) -> list[str]:
    out = []
    seen = set()
    for item in results:
        did = extract_doc_id(item)
        if did is None or did in seen:
            continue
        out.append(str(did))
        seen.add(did)
        if len(out) >= top_n:
            break
    return out


def reciprocal_rank(ranked_ids: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    for rank, did in enumerate(ranked_ids, start=1):
        if did in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    rel = sum(1 for did in top if did in relevant)
    return rel / k


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked_ids[:k]
    rel = sum(1 for did in top if did in relevant)
    return rel / len(relevant)


def evaluate_runs(
    runs: dict[str, dict[str, list[str]]],
    qrels: dict[str, set[str]],
    k_values: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_query_rows = []
    summary_rows = []

    for method, run in runs.items():
        rr_values = []
        p_values: dict[int, list[float]] = {k: [] for k in k_values}
        r_values: dict[int, list[float]] = {k: [] for k in k_values}

        qids = sorted(set(qrels.keys()) & set(run.keys()))
        for qid in qids:
            ranked = run[qid]
            rel = qrels.get(qid, set())
            row = {"method": method, "qid": qid}
            rr = reciprocal_rank(ranked, rel)
            row["MRR"] = rr
            rr_values.append(rr)

            for k in k_values:
                pk = precision_at_k(ranked, rel, k)
                rk = recall_at_k(ranked, rel, k)
                row[f"Precision@{k}"] = pk
                row[f"Recall@{k}"] = rk
                p_values[k].append(pk)
                r_values[k].append(rk)

            per_query_rows.append(row)

        summary = {
            "method": method,
            "queries_evaluated": len(qids),
            "MRR": float(statistics.fmean(rr_values)) if rr_values else 0.0,
        }
        for k in k_values:
            summary[f"Precision@{k}"] = float(statistics.fmean(p_values[k])) if p_values[k] else 0.0
            summary[f"Recall@{k}"] = float(statistics.fmean(r_values[k])) if r_values[k] else 0.0
        summary_rows.append(summary)

    per_query_df = pd.DataFrame(per_query_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("MRR", ascending=False)
    return per_query_df, summary_df


def inspect_project(cfg: Config) -> dict[str, Any]:
    checks = {
        "bm25_pickle": cfg.bm25_pickle.exists(),
        "dense_loader": cfg.dense_loader.exists(),
        "graphrag_loader": cfg.graphrag_loader.exists(),
        "qa_path": cfg.qa_path.exists(),
        "qrels_dir": cfg.qrels_dir.exists() and any(cfg.qrels_dir.glob("*.json")),
    }

    notes = []
    if not checks["qa_path"]:
        notes.append("QA benchmark file missing")
    if not checks["qrels_dir"]:
        notes.append("Qrels folder missing or empty")
    if not checks["bm25_pickle"]:
        notes.append("BM25 pickle missing (expected Step-1 output)")
    if not checks["dense_loader"]:
        notes.append("Dense loader missing (expected Step-1 output)")
    if not checks["graphrag_loader"]:
        notes.append("GraphRAG loader missing (expected Step-1 output)")

    return {
        "root": str(cfg.root),
        "paths": {
            "bm25_pickle": str(cfg.bm25_pickle),
            "dense_loader": str(cfg.dense_loader),
            "graphrag_loader": str(cfg.graphrag_loader),
            "qa_path": str(cfg.qa_path),
            "qrels_dir": str(cfg.qrels_dir),
        },
        "checks": checks,
        "notes": notes,
    }


def load_retrievers(cfg: Config) -> dict[str, Any]:
    with cfg.bm25_pickle.open("rb") as f:
        bm25 = pickle.load(f)

    dense_module = import_module_from_path("dense_loader_mod", cfg.dense_loader)
    if not hasattr(dense_module, "load_dense_fixed"):
        raise AttributeError("Dense loader must expose load_dense_fixed(device=None, k=...) ")
    dense = dense_module.load_dense_fixed(k=cfg.top_n)

    graph_module = import_module_from_path("graphrag_loader_mod", cfg.graphrag_loader)
    graph = GraphAdapter(graph_module)

    return {"bm25": bm25, "dense": dense, "graph": graph}


def build_methods(retrievers: dict[str, Any]) -> dict[str, Any]:
    bm25 = MethodAdapter(retrievers["bm25"], method="search")
    dense = MethodAdapter(retrievers["dense"], method="search")
    graph = MethodAdapter(retrievers["graph"], method="search")
    hybrid = HybridFusionRetriever(retrievers["bm25"], retrievers["dense"], retrievers["graph"], rerank=False)
    rerank = HybridFusionRetriever(retrievers["bm25"], retrievers["dense"], retrievers["graph"], rerank=True)

    return {
        "BM25": bm25,
        "Dense": dense,
        "GraphRAG": graph,
        "Hybrid": hybrid,
        "ReRank": rerank,
    }


def run_methods(cfg: Config, methods: dict[str, Any], qa_data: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    runs: dict[str, dict[str, list[str]]] = {}

    for method_name, retr in methods.items():
        run = defaultdict(list)
        print(f"\\nRunning {method_name}...")
        for item in qa_data:
            qid = str(item["id"])
            qtext = str(item[cfg.query_field])
            docs = retr.search(qtext, top_k=cfg.top_n)
            run[qid] = normalize_docs(docs, top_n=cfg.top_n)
        runs[method_name] = dict(run)
        print(f"  Queries processed: {len(run)}")

    return runs


def _doc_metadata(doc: Any) -> dict[str, Any]:
    if hasattr(doc, "metadata") and isinstance(getattr(doc, "metadata"), dict):
        return doc.metadata
    payload = getattr(doc, "__dict__", {})
    if isinstance(payload, dict):
        inner = payload.get("__dict__", {})
        if isinstance(inner, dict):
            meta = inner.get("metadata", {})
            if isinstance(meta, dict):
                return meta
    return {}


def build_runs_from_cached_rerank(cfg: Config, qids: list[str]) -> dict[str, dict[str, list[str]]]:
    if cfg.cached_rerank_dir is None or not cfg.cached_rerank_dir.exists():
        raise FileNotFoundError("Cached rerank directory not found.")

    ensure_fake_langchain_document_modules()
    runs = {m: {} for m in ["BM25", "Dense", "GraphRAG", "Hybrid", "ReRank"]}

    for qid in qids:
        fp = cfg.cached_rerank_dir / f"{qid}.pkl"
        if not fp.exists():
            continue
        docs = pickle.loads(fp.read_bytes())
        if not isinstance(docs, list):
            continue

        rows = []
        for idx, d in enumerate(docs):
            meta = _doc_metadata(d)
            did = meta.get("chunk_id") or meta.get("record_id") or meta.get("doc_id")
            if did is None:
                continue
            rows.append({
                "did": str(did),
                "rerank_pos": idx + 1,
                "bm25_rank": int(meta.get("bm25_rank", 10**9)),
                "dense_rank": int(meta.get("dense_rank", 10**9)),
                "grag_rank": int(meta.get("grag_rank", 10**9)),
                "hybrid_score": float(meta.get("hybrid_score", 0.0)),
            })

        if not rows:
            continue

        # ReRank: keep saved order from reranked candidates
        rerank_ids = [r["did"] for r in sorted(rows, key=lambda r: r["rerank_pos"])]
        bm25_ids = [r["did"] for r in sorted(rows, key=lambda r: (r["bm25_rank"], r["rerank_pos"]))]
        dense_ids = [r["did"] for r in sorted(rows, key=lambda r: (r["dense_rank"], r["rerank_pos"]))]
        graph_ids = [r["did"] for r in sorted(rows, key=lambda r: (r["grag_rank"], r["rerank_pos"]))]
        hybrid_ids = [r["did"] for r in sorted(rows, key=lambda r: (r["hybrid_score"], -r["rerank_pos"]), reverse=True)]

        runs["ReRank"][qid] = rerank_ids[: cfg.top_n]
        runs["BM25"][qid] = bm25_ids[: cfg.top_n]
        runs["Dense"][qid] = dense_ids[: cfg.top_n]
        runs["GraphRAG"][qid] = graph_ids[: cfg.top_n]
        runs["Hybrid"][qid] = hybrid_ids[: cfg.top_n]

    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce baseline retrieval metrics.")
    parser.add_argument("--root", type=Path, default=Path("baseline/advanced_genAI-main/data"))
    parser.add_argument("--bm25-pickle", type=Path, default=None)
    parser.add_argument("--dense-loader", type=Path, default=None)
    parser.add_argument("--graphrag-loader", type=Path, default=None)
    parser.add_argument("--qa-path", type=Path, default=None)
    parser.add_argument("--qrels-dir", type=Path, default=None)
    parser.add_argument("--cached-rerank-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("baseline/advanced_genAI-main/results/baseline_repro"))
    parser.add_argument("--query-field", type=str, default="question")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--k-values", type=str, default="1,3,5,10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    cfg = Config(
        root=root,
        bm25_pickle=(args.bm25_pickle or (root / "storage/subsample/retrieval_downstream/bm25_fixed_qe.pkl")).resolve(),
        dense_loader=(args.dense_loader or (root / "storage/subsample/vectordb_dense/load_dense_fixed.py")).resolve(),
        graphrag_loader=(args.graphrag_loader or (root / "storage/subsample/retrieval_graph/load_graphrag.py")).resolve(),
        qa_path=(args.qa_path or (root / "benchmark/benchmark_qa_bilingual.json")).resolve(),
        qrels_dir=(args.qrels_dir or (root / "benchmark/score/fixed_size")).resolve(),
        cached_rerank_dir=(args.cached_rerank_dir or (root / "storage/subsample/hybrid/candidates_rerank")).resolve(),
        output_dir=args.output_dir.resolve(),
        seed=args.seed,
        k_values=tuple(int(k.strip()) for k in args.k_values.split(",") if k.strip()),
        top_n=args.top_n,
        query_field=args.query_field,
    )

    set_seed(cfg.seed)

    inspection = inspect_project(cfg)
    print("=== Baseline Repro Inspection ===")
    print(json.dumps(inspection, indent=2))

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "inspection.json").write_text(json.dumps(inspection, indent=2), encoding="utf-8")

    if args.inspect_only:
        return

    missing = [k for k, ok in inspection["checks"].items() if not ok]
    if missing:
        raise FileNotFoundError(
            "Cannot run baselines. Missing required assets: " + ", ".join(missing)
        )

    qa_data = json.loads(cfg.qa_path.read_text(encoding="utf-8"))
    qrels = load_qrels(cfg.qrels_dir)

    try:
        retrievers = load_retrievers(cfg)
        methods = build_methods(retrievers)
        runs = run_methods(cfg, methods, qa_data)
    except Exception as e:
        print(f"WARNING: live retriever execution unavailable ({type(e).__name__}: {e})")
        print("Falling back to cached rerank candidates for method reconstruction.")
        qids = [str(x["id"]) for x in qa_data]
        runs = build_runs_from_cached_rerank(cfg, qids)

    per_query_df, summary_df = evaluate_runs(runs, qrels, cfg.k_values)

    # Save comparable artifacts
    (cfg.output_dir / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    per_query_df.to_csv(cfg.output_dir / "metrics_per_query.csv", index=False)
    summary_df.to_csv(cfg.output_dir / "metrics_summary.csv", index=False)

    print("\n=== Summary Metrics ===")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df.round(4).to_string(index=False))

    print(f"\nSaved artifacts to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
