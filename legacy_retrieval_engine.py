"""Helper module for the reproduced baseline retrieval project.

This file contains the reusable helpers from the reproduced baseline work.
It includes: retrieval adapters, Confidence/Waterfall/Voting orchestration,
and wrapper functions returning `(answer, docs, trace)`.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

try:
    import nltk
    from langdetect import detect
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
except ImportError as exc:  # fail with a clearer message when notebook deps are missing
    raise ImportError(
        "legacy_retrieval_engine.py requires the legacy RAG dependencies. "
        "Install the notebook requirements before importing this module."
    ) from exc

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

for package in ("punkt", "stopwords", "punkt_tab"):
    nltk.download(package, quiet=True)

STOP_EN = set(nltk.corpus.stopwords.words("english"))
STOP_DE = set(nltk.corpus.stopwords.words("german"))

EVAL_SCOPE = "full_corpus"
K_VALUES = (1, 3, 5, 10)
RETRIEVE_K = 50
TOP_K = 10

WEIGHT_PRESETS = {
    "entity_temporal": {"bm25": 0.7, "dense": 1.1, "graph": 1.4},
    "entity": {"bm25": 0.9, "dense": 1.1, "graph": 1.3},
    "keyword": {"bm25": 1.3, "dense": 1.1, "graph": 0.7},
    "semantic": {"bm25": 0.8, "dense": 1.4, "graph": 0.9},
    "graph": {"bm25": 0.8, "dense": 1.0, "graph": 1.4},
    "mixed": {"bm25": 1.0, "dense": 1.2, "graph": 0.8},
}

GATE_THRESHOLD = 0.75
USE_BILINGUAL_QUERY = False
USE_GATING = True
USE_RETRY = True

CANDIDATE_ROOTS = [
    pathlib.Path.cwd(),
    pathlib.Path.cwd() / "baseline/advanced_genAI-main/data",
    pathlib.Path(__file__).resolve().parent,
    pathlib.Path(__file__).resolve().parent / "baseline/advanced_genAI-main/data",
    pathlib.Path("/content/advanced-genai-26/baseline/advanced_genAI-main/data"),
    pathlib.Path("/content/drive/MyDrive/Adv_GenAI"),
    pathlib.Path("/content/drive/MyDrive/advanced-genai-26/baseline/advanced_genAI-main/data"),
    pathlib.Path("/content/drive/MyDrive/advanced_genAI-main/data"),
]


def looks_like_project_root(path: pathlib.Path) -> bool:
    return (path / "benchmark").exists() and (path / "storage").exists()


def find_project_root() -> pathlib.Path:
    for candidate in CANDIDATE_ROOTS:
        if looks_like_project_root(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not auto-detect project root containing benchmark/ and storage/. "
        "Run from the repository root or set up the expected Colab data path."
    )


PROJECT_ROOT = find_project_root()
PATH_BM25_PICKLE = PROJECT_ROOT / "storage/full_corpus/retrieval/fixed_size_chunk/bm25_retriever_full.pkl"
PATH_DENSE_INDEX = PROJECT_ROOT / "storage/full_corpus/vectordb_dense/fixed_e5"
PATH_GRAG_ROOT = PROJECT_ROOT / "storage/full_corpus/retrieval_graph"
PATH_CHUNK_PKL = PROJECT_ROOT / "storage/full_corpus/Lang_norm/fixed_size_chunk/docs_fixed_norm.pkl"

for required_path in (PATH_BM25_PICKLE, PATH_DENSE_INDEX, PATH_GRAG_ROOT, PATH_CHUNK_PKL):
    if not required_path.exists():
        raise FileNotFoundError(f"Missing required retrieval artifact: {required_path}")


class BilingualBM25:
    """Compatibility class for notebook pickles."""

    def _rank_lang(self, query: str, lang: str, top_k: int):
        try:
            query_tokens = nltk.word_tokenize(query)
        except Exception:
            query_tokens = query.split()
        scores = self.bm25[lang].get_scores(query_tokens)
        idx = np.argsort(scores)[::-1][:top_k]
        hits = []
        for item_idx in idx:
            doc = self.docs_by_lang[lang][item_idx]
            doc.metadata["bm25_score"] = float(scores[item_idx])
            hits.append(doc)
        return hits

    def _get_docs_with_scores(self, retriever, query, top_k):
        if hasattr(retriever, "get_relevant_documents_with_scores"):
            try:
                return retriever.get_relevant_documents_with_scores(query, k=top_k)
            except Exception:
                pass

        if hasattr(retriever, "vectorizer") and hasattr(retriever, "docs"):
            try:
                tokens = query.lower().split()
                scores = retriever.vectorizer.get_scores(tokens)
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
                return [(retriever.docs[idx], float(score)) for idx, score in ranked]
            except Exception:
                pass

        if hasattr(retriever, "invoke"):
            try:
                old_k = getattr(retriever, "k", None)
                if old_k is not None:
                    retriever.k = top_k
                docs = retriever.invoke(query)
                if old_k is not None:
                    retriever.k = old_k
                return [(doc, doc.metadata.get("score", 0.0)) for doc in docs[:top_k]]
            except Exception:
                pass

        return []

    def search(self, query: str, top_k: int = 100):
        if hasattr(self, "bm25") and hasattr(self, "docs_by_lang"):
            src = detect(query) if query.strip() else "en"
            src = src if src in ("en", "de") else "en"
            bag = []
            translator = getattr(self, "translator", None)
            for lang in ("en", "de"):
                query_lang = translator.translate(query, lang) if translator and lang != src else query
                bag.extend(self._rank_lang(query_lang, lang, top_k))

            best = {}
            for doc in bag:
                uid = doc.metadata.get("chunk_id") or doc.metadata.get("record_id")
                if uid not in best or doc.metadata["bm25_score"] > best[uid].metadata.get("bm25_score", -1e9):
                    best[uid] = doc
            return sorted(best.values(), key=lambda doc: doc.metadata.get("bm25_score", 0.0), reverse=True)[:top_k]

        if hasattr(self, "retrievers") and isinstance(self.retrievers, dict):
            src = detect(query) if query.strip() else "en"
            src = src if src in ("en", "de") else "en"
            bag = []
            translator = getattr(self, "translator", None)

            for lang, retriever in self.retrievers.items():
                query_lang = translator.translate(query, lang) if translator and lang != src else query
                docs_with_scores = self._get_docs_with_scores(retriever, query_lang, top_k)
                for doc, score in docs_with_scores:
                    doc.metadata["bm25_score"] = float(score)
                    bag.append(doc)

            best = {}
            for doc in bag:
                uid = doc.metadata.get("chunk_id") or doc.metadata.get("record_id")
                if uid is None:
                    continue
                if uid not in best or doc.metadata.get("bm25_score", -1e9) > best[uid].metadata.get("bm25_score", -1e9):
                    best[uid] = doc

            return sorted(best.values(), key=lambda doc: doc.metadata.get("bm25_score", 0.0), reverse=True)[:top_k]

        raise AttributeError("Unsupported BilingualBM25 object format.")


class QEBM25:
    @staticmethod
    def _expand_query(query: str, base_retriever, fb_docs: int = 5, fb_terms: int = 5) -> str:
        def tokenize(text: str):
            try:
                return nltk.word_tokenize(text.lower())
            except Exception:
                return text.lower().split()

        hits = base_retriever.search(query, top_k=fb_docs)
        tokens = [
            token
            for hit in hits
            for token in tokenize(hit.page_content)
            if token.isalpha() and token not in STOP_EN and token not in STOP_DE
        ]
        extra = " ".join(word for word, _ in nltk.FreqDist(tokens).most_common(fb_terms))
        return f"{query} {extra}" if extra else query

    def search(self, query: str, top_k: int = 100):
        if hasattr(self, "base"):
            expanded = self._expand_query(query, self.base)
            return self.base.search(expanded, top_k)
        raise AttributeError("QEBM25 object missing base retriever.")


class BM25RetrieverAdapter:
    def __init__(self, obj):
        self.obj = obj

    def search(self, query: str, top_k: int = 100):
        if hasattr(self.obj, "search"):
            try:
                return self.obj.search(query, top_k=top_k)
            except TypeError:
                return self.obj.search(query, k=top_k)

        if hasattr(self.obj, "invoke"):
            old_k = getattr(self.obj, "k", None)
            if old_k is not None:
                self.obj.k = top_k
            docs = self.obj.invoke(query)
            if old_k is not None:
                self.obj.k = old_k
            for rank, doc in enumerate(docs, start=1):
                if hasattr(doc, "metadata"):
                    doc.metadata.setdefault("bm25_score", float(top_k - rank))
            return docs[:top_k]

        raise AttributeError(f"Unsupported BM25 object type: {type(self.obj)}")


main_module = sys.modules.get("__main__")
if main_module is not None:
    setattr(main_module, "BilingualBM25", BilingualBM25)
    setattr(main_module, "QEBM25", QEBM25)


class DenseRetriever:
    def __init__(self, index_dir: pathlib.Path, model_name="intfloat/multilingual-e5-large-instruct", k: int = 100):
        self.k = k
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if os.path.exists("/proc/driver/nvidia/version") else "cpu"},
            encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
        )
        self.store = Chroma(persist_directory=str(index_dir), embedding_function=self.embeddings)

    def _prep(self, query: str) -> str:
        return "query: " + query.strip()

    def search(self, query: str, top_k: int = 100):
        k = top_k or self.k
        hits = self.store.similarity_search_with_score(self._prep(query), k=k)
        out = []
        for doc, dist in hits:
            doc.metadata["dense_score"] = 1.0 - float(dist)
            out.append(doc)
        return out


class GraphRAGRetriever:
    def __init__(self, graph_root: pathlib.Path, chunk_pkl: pathlib.Path):
        self.root = graph_root
        self.emb_dir = graph_root / "embeddings"
        self.chunk_pkl = chunk_pkl
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._emb_cache = {}
        self._cid_cache = {}
        self._chunk_by_id = None
        self._chunk_vec_cache = {}
        self.comm2chunk = json.loads((self.root / "comm2chunk_fixed.json").read_text(encoding="utf-8"))
        self._available_levels = []
        if self.emb_dir.exists():
            for path in sorted(self.emb_dir.glob("EMB_fixed_C*.npy")):
                try:
                    self._available_levels.append(int(path.stem.split("C")[-1]))
                except ValueError:
                    continue
        self._default_level = self._available_levels[0] if self._available_levels else 1

    def _load_embeddings(self, level: int):
        if level in self._emb_cache:
            return self._emb_cache[level], self._cid_cache[level]
        matrix = np.load(self.emb_dir / f"EMB_fixed_C{level}.npy")
        community_ids = json.loads((self.emb_dir / f"CID_fixed_C{level}.json").read_text(encoding="utf-8"))
        self._emb_cache[level] = matrix
        self._cid_cache[level] = community_ids
        return matrix, community_ids

    def _load_chunks(self):
        if self._chunk_by_id is not None:
            return self._chunk_by_id
        with open(self.chunk_pkl, "rb") as handle:
            docs_norm = pickle.load(handle)

        def restore(doc):
            raw = doc.metadata.get("original_text") or doc.page_content
            return Document(page_content=raw, metadata=doc.metadata)

        docs = [restore(doc) for doc in docs_norm]
        self._chunk_by_id = {doc.metadata["chunk_id"]: doc for doc in docs}
        return self._chunk_by_id

    def _chunk_vec(self, chunk_id: str, chunks: dict):
        if chunk_id not in self._chunk_vec_cache:
            self._chunk_vec_cache[chunk_id] = self.embedder.encode(
                [chunks[chunk_id].page_content], normalize_embeddings=True
            )[0]
        return self._chunk_vec_cache[chunk_id]

    def retrieve(self, query: str, level: str = None, k_comms: int = 24, top_k: int = 100):
        level_num = self._default_level if level is None else int(level.lstrip("C"))
        emb_mat, cid_list = self._load_embeddings(level_num)
        chunks = self._load_chunks()

        query_vec = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims_comm = emb_mat @ query_vec
        best_idx = sims_comm.argsort()[::-1][:k_comms]

        candidate_ids = set()
        for idx in best_idx:
            candidate_ids.update(self.comm2chunk.get(cid_list[idx], []))

        scored = []
        for chunk_id in candidate_ids:
            if chunk_id not in chunks:
                continue
            sim = float(self._chunk_vec(chunk_id, chunks) @ query_vec)
            scored.append((chunk_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        out = []
        for chunk_id, sim in scored[:top_k]:
            doc = chunks[chunk_id]
            doc.metadata["grag_score"] = (sim + 1.0) / 2.0
            out.append(doc)
        return out

    def search(self, query: str, top_k: int = 100, k_comms: int = 48):
        return self.retrieve(query=query, level=None, k_comms=k_comms, top_k=top_k)


def _uid(doc: Any):
    meta = getattr(doc, "metadata", {}) or {}
    return meta.get("chunk_id") or meta.get("record_id") or meta.get("doc_id")


def _safe_unique(docs):
    out, seen = [], set()
    for doc in docs:
        uid = _uid(doc)
        if uid is None or uid in seen:
            continue
        seen.add(uid)
        out.append(doc)
    return out


def _rrf_fuse(runs: dict, k_rrf: int = 60, weights=None):
    weights = weights or {"bm25": 1.0, "dense": 1.2, "graph": 0.8}
    scores = defaultdict(float)
    store = {}
    for name, docs in runs.items():
        weight = float(weights.get(name, 1.0))
        for rank, doc in enumerate(docs, start=1):
            uid = _uid(doc)
            if uid is None:
                continue
            store.setdefault(uid, doc)
            scores[uid] += weight * (1.0 / (k_rrf + rank))
    fused = sorted(store.values(), key=lambda doc: scores[_uid(doc)], reverse=True)
    for doc in fused:
        doc.metadata["fused_score"] = float(scores[_uid(doc)])
    return fused


def _token_set(text: str):
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return {token for token in text.split() if token}


def _overlap_rerank(docs, query: str, top_k: int):
    query_terms = {token.lower() for token in query.split() if token.strip()}
    scored = []
    for doc in docs:
        text = (doc.metadata.get("original_text") or doc.page_content or "").lower()
        overlap = len(query_terms & set(text.split())) / max(len(query_terms), 1)
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


@dataclass
class AgentState:
    query: str
    normalized_query: str = ""
    query_type: str = "mixed"
    query_hints: Dict[str, float] = field(default_factory=dict)
    retrieval_by_agent: Dict[str, List[Any]] = field(default_factory=dict)
    fused_docs: List[Any] = field(default_factory=list)
    reranked_docs: List[Any] = field(default_factory=list)
    final_answer: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    critic_ok: bool = False
    critic_feedback: str = ""
    needs_reretrieval: bool = False


class BaseAgent:
    name = "base"

    def run(self, state: AgentState, **kwargs) -> AgentState:
        raise NotImplementedError


class QueryUnderstandingAgent(BaseAgent):
    name = "query_understanding"
    _WHO_ROLE_RE = re.compile(r"^who\b", re.I)
    _YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")

    def run(self, state: AgentState, **kwargs) -> AgentState:
        query = (state.query or "").strip()
        query_low = query.lower()
        state.normalized_query = " ".join(query.split())

        graph_signals = {"relationship", "connected", "connection", "connections", "dependency", "impact", "between"}
        keyword_signals = {"exactly", "define", "list", "when", "where", "who"}

        graph_score = sum(1 for term in graph_signals if term in query_low)
        keyword_score = sum(1 for term in keyword_signals if term in query_low)
        has_year = bool(self._YEAR_RE.search(query))
        who_role_query = bool(self._WHO_ROLE_RE.match(query))
        factual_starts = (
            "when ", "where ", "which ", "what year", "what date", "what is ", "what are ",
            "what was ", "what were ", "what does ", "what do ", "what did ",
        )
        factual_query = query_low.startswith(factual_starts)

        if who_role_query and has_year:
            query_type = "entity_temporal"
        elif who_role_query:
            query_type = "entity"
        elif graph_score >= 1:
            query_type = "graph"
        elif factual_query or keyword_score >= 1:
            query_type = "keyword"
        elif len(query.split()) >= 7:
            query_type = "semantic"
        else:
            query_type = "mixed"

        state.query_type = query_type
        match = self._YEAR_RE.search(query)
        state.query_hints["_year"] = int(match.group()) if match else None
        state.query_hints.update(WEIGHT_PRESETS.get(query_type, WEIGHT_PRESETS["mixed"]))
        return state


class BM25RetrieverAgent(BaseAgent):
    name = "bm25_retriever"

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state: AgentState, top_k: int = 30, **kwargs) -> AgentState:
        state.retrieval_by_agent["bm25"] = _safe_unique(self.retriever.search(state.normalized_query, top_k=top_k))
        return state


class DenseRetrieverAgent(BaseAgent):
    name = "dense_retriever"

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state: AgentState, top_k: int = 30, **kwargs) -> AgentState:
        state.retrieval_by_agent["dense"] = _safe_unique(self.retriever.search(state.normalized_query, top_k=top_k))
        return state


class GraphRetrieverAgent(BaseAgent):
    name = "graph_retriever"

    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state: AgentState, top_k: int = 30, **kwargs) -> AgentState:
        state.retrieval_by_agent["graph"] = _safe_unique(
            self.retriever.search(state.normalized_query, top_k=top_k, k_comms=48)
        )
        return state


class FusionAgent(BaseAgent):
    name = "fusion"

    def run(self, state: AgentState, top_k: int = 30, **kwargs) -> AgentState:
        runs = {
            "bm25": state.retrieval_by_agent.get("bm25", []),
            "dense": state.retrieval_by_agent.get("dense", []),
            "graph": state.retrieval_by_agent.get("graph", []),
        }
        fused = _rrf_fuse(runs, weights=state.query_hints or None)
        state.fused_docs = _safe_unique(fused)[:top_k]
        return state


class ReRankerAgent(BaseAgent):
    name = "reranker"
    _ce_model = None

    @classmethod
    def _get_cross_encoder(cls):
        if cls._ce_model is None:
            cls._ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return cls._ce_model

    def run(self, state: AgentState, top_k: int = 10, **kwargs) -> AgentState:
        docs = state.fused_docs
        if not docs:
            state.reranked_docs = []
            return state

        query = state.normalized_query
        candidates = docs[:50]
        try:
            model = self._get_cross_encoder()
            pairs = [
                (query, (doc.metadata.get("original_text") or doc.page_content or "").strip()[:512])
                for doc in candidates
            ]
            scores = model.predict(pairs)
            ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
            for doc, score in ranked:
                doc.metadata["rerank_score"] = float(score)
            state.reranked_docs = [doc for doc, _ in ranked[:top_k]]
        except Exception:
            state.reranked_docs = _overlap_rerank(docs, query, top_k=top_k)
            for doc in state.reranked_docs:
                doc.metadata.setdefault("rerank_score", 0.0)
        return state


class AnswerSynthesizerAgent(BaseAgent):
    name = "answer_synthesizer"

    def _build_context(self, docs: List[Any], max_docs: int = 5) -> str:
        parts = []
        for doc in docs[:max_docs]:
            text = (doc.metadata.get("original_text") or doc.page_content or "").strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def run(self, state: AgentState, **kwargs) -> AgentState:
        context = self._build_context(state.reranked_docs or state.fused_docs)
        query = state.normalized_query
        year = state.query_hints.get("_year")

        if not context:
            state.final_answer = "No supporting context was retrieved."
            state.evidence_ids = []
            return state

        sentences = re.split(r"(?<=[.!?])\s+", context)
        query_terms = set(_token_set(query)) - STOP_EN - STOP_DE
        scored = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            sentence_terms = set(_token_set(sentence))
            base = len(query_terms & sentence_terms) / max(len(sentence_terms), 1) * len(query_terms & sentence_terms)
            if year and re.search(r"\b" + str(year) + r"\b", sentence):
                base *= 2.5
            elif year and any(re.search(r"\b" + str(item) + r"\b", sentence) for item in [year - 1, year + 1, year - 2, year + 2]):
                base *= 1.4
            scored.append((base, sentence))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = []
        seen_terms = set()
        for score, sentence in scored:
            sentence_terms = set(_token_set(sentence))
            if seen_terms and len(sentence_terms & seen_terms) / max(len(sentence_terms), 1) > 0.6:
                continue
            best.append(sentence)
            seen_terms |= sentence_terms
            if len(best) >= 3:
                break

        if not best or scored[0][0] == 0:
            state.final_answer = "The retrieved context does not contain information about the query" + (
                f" from {year}" if year else ""
            ) + "."
        else:
            state.final_answer = " ".join(best)

        state.evidence_ids = [_uid(doc) for doc in (state.reranked_docs or state.fused_docs)[:5] if _uid(doc) is not None]
        return state


class CriticAgent(BaseAgent):
    name = "critic"
    _YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")

    def _temporal_coherent(self, answer: str, query_year: int, window: int = 10) -> bool:
        found = [int(match) for match in self._YEAR_RE.findall(answer)]
        return any(abs(year - query_year) <= window for year in found)

    def run(self, state: AgentState, min_support_overlap: float = 0.45, **kwargs) -> AgentState:
        answer_terms = _token_set(state.final_answer) - STOP_EN - STOP_DE
        if not answer_terms:
            state.critic_ok = False
            state.needs_reretrieval = True
            state.critic_feedback = "Answer is empty."
            return state

        support_docs = state.reranked_docs or state.fused_docs
        if not support_docs:
            state.critic_ok = False
            state.needs_reretrieval = True
            state.critic_feedback = "No support documents available."
            return state

        support_text = " ".join((doc.metadata.get("original_text") or doc.page_content or "") for doc in support_docs[:5])
        support_terms = _token_set(support_text) - STOP_EN - STOP_DE
        overlap = len(answer_terms & support_terms) / max(len(answer_terms), 1)

        per_doc_max = 0.0
        for doc in support_docs[:5]:
            doc_terms = _token_set(doc.metadata.get("original_text") or doc.page_content or "") - STOP_EN - STOP_DE
            per_doc_max = max(per_doc_max, len(answer_terms & doc_terms) / max(len(answer_terms), 1))

        grounded = overlap >= min_support_overlap and per_doc_max >= 0.25
        query_year = state.query_hints.get("_year")
        temporal_ok = True
        temporal_msg = ""
        if state.query_type == "entity_temporal" and query_year:
            temporal_ok = self._temporal_coherent(state.final_answer, query_year, window=10)
            temporal_msg = f" Temporal check (target={query_year}±10): " + (
                "PASS" if temporal_ok else "FAIL — answer references wrong time period"
            ) + "."

        state.critic_ok = grounded and temporal_ok
        state.needs_reretrieval = not state.critic_ok
        state.critic_feedback = (
            f"Global overlap={overlap:.3f}, best-doc overlap={per_doc_max:.3f}; "
            f"threshold={min_support_overlap:.2f}/0.25."
            + temporal_msg
            + (" Grounded." if state.critic_ok else " Potentially ungrounded: trigger re-retrieval.")
        )
        return state


DE_EN_TERM_MAP = {
    "rektor": "rector", "rektorin": "rector", "praesident": "president", "präsident": "president",
    "forschung": "research", "studierende": "students", "professor": "professor", "institut": "institute",
    "jahr": "year", "zwischen": "between", "wer": "who", "wann": "when",
}
EN_DE_TERM_MAP = {
    "rector": "rektor", "president": "präsident", "research": "forschung", "students": "studierende",
    "between": "zwischen", "year": "jahr", "who": "wer", "when": "wann",
}


def make_query_variants(query: str, enable: bool = USE_BILINGUAL_QUERY):
    if not enable or not query:
        return [query]
    query_low = query.lower()
    has_de = any(re.search(r"\b" + term + r"\b", query_low) for term in DE_EN_TERM_MAP)
    has_en = any(re.search(r"\b" + term + r"\b", query_low) for term in EN_DE_TERM_MAP)
    variants = [query]
    if has_de and not has_en:
        expanded = query_low
        for de_term, en_term in DE_EN_TERM_MAP.items():
            expanded = re.sub(r"\b" + de_term + r"\b", en_term, expanded)
        if expanded != query_low:
            variants.append(expanded)
    elif has_en and not has_de:
        expanded = query_low
        for en_term, de_term in EN_DE_TERM_MAP.items():
            expanded = re.sub(r"\b" + en_term + r"\b", de_term, expanded)
        if expanded != query_low:
            variants.append(expanded)
    return variants


class ConfidenceOrchestrator:
    def __init__(
        self,
        weight_presets=None,
        gate_threshold: float = GATE_THRESHOLD,
        use_gating: bool = USE_GATING,
        use_retry: bool = USE_RETRY,
        use_bilingual: bool = USE_BILINGUAL_QUERY,
    ):
        self.weight_presets = weight_presets or WEIGHT_PRESETS
        self.gate_threshold = gate_threshold
        self.use_gating = use_gating
        self.use_retry = use_retry
        self.use_bilingual = use_bilingual
        self.query_agent = QueryUnderstandingAgent()
        self.retrievers = {
            "bm25": BM25RetrieverAgent(bm25_retriever),
            "dense": DenseRetrieverAgent(dense_retriever),
            "graph": GraphRetrieverAgent(graph_retriever),
        }
        self.fusion_agent = FusionAgent()
        self.reranker_agent = ReRankerAgent()
        self.answer_agent = AnswerSynthesizerAgent()
        self.critic_agent = CriticAgent()

    def _select_preset(self, query_type: str):
        return dict(self.weight_presets.get(query_type, self.weight_presets["mixed"]))

    def _apply_gating(self, weights: dict):
        if not self.use_gating:
            return dict(weights), []
        gated = [name for name, weight in weights.items() if weight < self.gate_threshold]
        kept = {name: weight for name, weight in weights.items() if weight >= self.gate_threshold}
        if not kept:
            return dict(weights), []
        return kept, gated

    def _retrieve(self, state: AgentState, active_names, retrieve_k: int, query_variants):
        if not hasattr(state, "retrieval_errors"):
            state.retrieval_errors = {}
        if not hasattr(state, "zero_result_retrievers"):
            state.zero_result_retrievers = []

        for name in active_names:
            agent = self.retrievers[name]
            merged, seen = [], set()
            had_error = None
            for variant in query_variants:
                state.normalized_query = variant
                try:
                    agent.run(state, top_k=retrieve_k)
                except Exception as exc:
                    had_error = repr(exc)
                    state.retrieval_by_agent[name] = []
                    continue
                for doc in state.retrieval_by_agent.get(name, []):
                    uid = _uid(doc)
                    if uid and uid not in seen:
                        seen.add(uid)
                        merged.append(doc)
            state.retrieval_by_agent[name] = merged
            if merged:
                state.retrieval_errors.pop(name, None)
                if name in state.zero_result_retrievers:
                    state.zero_result_retrievers.remove(name)
            else:
                if name not in state.zero_result_retrievers:
                    state.zero_result_retrievers.append(name)
                state.retrieval_errors[name] = had_error or "retriever returned 0 unique docs"

        for name in self.retrievers:
            state.retrieval_by_agent.setdefault(name, [])
        return state

    def run(self, query: str, retrieve_k: int = RETRIEVE_K, top_k: int = TOP_K):
        start = time.time()
        state = AgentState(query=query)
        state.retrieval_errors = {}
        state.zero_result_retrievers = []
        state = self.query_agent.run(state)
        original_norm = state.normalized_query
        preset_weights = self._select_preset(state.query_type)
        active_weights, gated_out = self._apply_gating(preset_weights)
        variants = make_query_variants(original_norm, enable=self.use_bilingual)

        state = self._retrieve(state, list(active_weights.keys()), retrieve_k, variants)
        state.normalized_query = original_norm
        state.query_hints = {**state.query_hints, **active_weights}
        state = self.fusion_agent.run(state, top_k=retrieve_k)
        state = self.reranker_agent.run(state, top_k=top_k)
        state = self.answer_agent.run(state)
        state = self.critic_agent.run(state)

        retry_triggered = False
        retry_weights = None
        if self.use_retry and state.needs_reretrieval:
            retry_triggered = True
            broadened = {name: max(preset_weights.get(name, 1.0), 1.0) for name in self.retrievers}
            retry_weights = dict(broadened)
            state = self._retrieve(state, list(broadened.keys()), retrieve_k, variants)
            state.query_hints = {**state.query_hints, **broadened}
            state = self.fusion_agent.run(state, top_k=retrieve_k)
            state = self.reranker_agent.run(state, top_k=top_k)
            state = self.answer_agent.run(state)
            state = self.critic_agent.run(state)

        trace = {
            "query": query,
            "query_type": state.query_type,
            "weights": active_weights,
            "retry_weights": retry_weights,
            "gated_out": gated_out,
            "retriever_counts": {name: len(state.retrieval_by_agent.get(name, [])) for name in self.retrievers},
            "retrieval_errors": dict(getattr(state, "retrieval_errors", {})),
            "zero_result_retrievers": list(getattr(state, "zero_result_retrievers", [])),
            "retry_triggered": retry_triggered,
            "critic_ok": state.critic_ok,
            "critic_feedback": state.critic_feedback,
            "evidence_ids": state.evidence_ids,
            "latency_s": round(time.time() - start, 4),
            "bilingual_variants": len(variants),
        }
        top_docs = (state.reranked_docs or state.fused_docs)[:top_k]
        return state.final_answer, top_docs, trace


class WaterfallOrchestrator(ConfidenceOrchestrator):
    def run(self, query: str, retrieve_k: int = RETRIEVE_K, top_k: int = TOP_K):
        start = time.time()
        state = AgentState(query=query)
        state.retrieval_errors = {}
        state.zero_result_retrievers = []
        state = self.query_agent.run(state)
        original_norm = state.normalized_query
        tiers = [
            (["bm25"], {"bm25": 1.0}),
            (["bm25", "dense"], {"bm25": 1.0, "dense": 1.2}),
            (["bm25", "dense", "graph"], {"bm25": 1.0, "dense": 1.2, "graph": 1.4}),
        ]
        trace = {"query": query, "query_type": state.query_type, "tiers_attempted": 0, "tier_results": []}

        final_tier = 0
        for final_tier, (active_names, weights) in enumerate(tiers, 1):
            trace["tiers_attempted"] = final_tier
            state = self._retrieve(state, active_names, retrieve_k, [original_norm])
            state.normalized_query = original_norm
            state.query_hints = {**state.query_hints, **weights}
            state = self.fusion_agent.run(state, top_k=retrieve_k)
            state = self.reranker_agent.run(state, top_k=top_k)
            state = self.answer_agent.run(state)
            state = self.critic_agent.run(state)
            trace["tier_results"].append({
                "tier": final_tier,
                "active": list(active_names),
                "weights": dict(weights),
                "critic_ok": state.critic_ok,
                "retriever_counts": {name: len(state.retrieval_by_agent.get(name, [])) for name in self.retrievers},
                "retrieval_errors": dict(getattr(state, "retrieval_errors", {})),
                "zero_result_retrievers": list(getattr(state, "zero_result_retrievers", [])),
            })
            if state.critic_ok:
                break

        if not state.critic_ok:
            year_hint = state.query_hints.get("_year")
            year_str = f" from {year_hint}" if year_hint else ""
            state.final_answer = (
                "The available corpus does not contain sufficient evidence to "
                f"reliably answer this query{year_str}. Retrieved context covers "
                "related topics but does not directly address the question."
            )

        trace.update({
            "latency_s": round(time.time() - start, 4),
            "final_tier": final_tier,
            "critic_ok": state.critic_ok,
            "critic_feedback": state.critic_feedback,
            "retriever_counts": {name: len(state.retrieval_by_agent.get(name, [])) for name in self.retrievers},
            "retrieval_errors": dict(getattr(state, "retrieval_errors", {})),
            "zero_result_retrievers": list(getattr(state, "zero_result_retrievers", [])),
            "evidence_ids": state.evidence_ids,
        })
        top_docs = (state.reranked_docs or state.fused_docs)[:top_k]
        return state.final_answer, top_docs, trace


with open(PATH_BM25_PICKLE, "rb") as handle:
    bm25_raw = pickle.load(handle)

bm25_retriever = BM25RetrieverAdapter(bm25_raw)
dense_retriever = DenseRetriever(PATH_DENSE_INDEX, k=100)
graph_retriever = GraphRAGRetriever(PATH_GRAG_ROOT, PATH_CHUNK_PKL)

orchestrator = ConfidenceOrchestrator()
waterfall_orchestrator = WaterfallOrchestrator()
voting_weight_presets = {key: {"bm25": 1.0, "dense": 1.0, "graph": 1.0} for key in WEIGHT_PRESETS}
voting_orchestrator = ConfidenceOrchestrator(
    weight_presets=voting_weight_presets,
    gate_threshold=0.0,
    use_gating=False,
    use_retry=False,
    use_bilingual=USE_BILINGUAL_QUERY,
)


def confidence_orchestrate(query, top_k=5):
    answer, docs, trace = orchestrator.run(query, top_k=top_k)
    return answer, docs, trace


def waterfall_orchestrate(query, top_k=5):
    answer, docs, trace = waterfall_orchestrator.run(query, top_k=top_k)
    return answer, docs, trace


def voting_orchestrate(query, top_k=5):
    answer, docs, trace = voting_orchestrator.run(query, top_k=top_k)
    return answer, docs, trace


__all__ = [
    "ConfidenceOrchestrator",
    "WaterfallOrchestrator",
    "orchestrator",
    "waterfall_orchestrator",
    "voting_orchestrator",
    "confidence_orchestrate",
    "waterfall_orchestrate",
    "voting_orchestrate",
    "RETRIEVE_K",
    "TOP_K",
    "K_VALUES",
    "PROJECT_ROOT",
]
