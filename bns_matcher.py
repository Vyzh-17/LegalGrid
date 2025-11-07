# bns_matcher.py
import os
import json
import re
import math
import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# heavy dependencies try-import
try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer, util
    SBER_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    util = None
    SBER_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    CROSS_AVAILABLE = True
except Exception:
    CrossEncoder = None
    CROSS_AVAILABLE = False

logger = logging.getLogger("BNSMatcher")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


def clean_text(s: str) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).replace("\r", " ").replace("\n", " ")).strip()


def alnum_lower(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower())


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-float(x))) if x > -700 else 0.0


def top_k_indices(array: np.ndarray, k: int):
    return list(np.argsort(-array)[:k])


class BNSMatcher:
    def __init__(
        self,
        dataset_path: str = "bns_dataset.csv",
        dictionary_path: str = "bns_dictionary.json",
        embeddings_path: str = "bns_embeddings.pt",
        embed_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        cross_encoder_model: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)
        self.df = pd.read_csv(dataset_path, dtype=str).fillna("")
        self.sections = []
        for _, row in self.df.iterrows():
            sec = clean_text(row.get("Section") or row.get("section") or row.get("Chapter") or "")
            title = clean_text(row.get("Section _name") or row.get("Section_name") or row.get("title") or "")
            desc = clean_text(row.get("Description") or row.get("description") or "")
            text = f"{title}\n\n{desc}".strip()
            self.sections.append({"section": sec, "title": title, "description": desc, "text": text})

        # dictionary (section -> [keywords])
        if os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, "r", encoding="utf-8") as f:
                    self.dictionary = json.load(f)
            except Exception:
                logger.exception("Failed to load dictionary.json; using empty dictionary")
                self.dictionary = {}
        else:
            self.dictionary = {}

        # choose device
        if device is None:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.device = device
        logger.info("Using device: %s", self.device)

        if not SBER_AVAILABLE:
            raise RuntimeError("Please install sentence-transformers (pip install sentence-transformers)")

        logger.info("Loading embedder: %s", embed_model_name)
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)

        self.embeddings_path = embeddings_path
        self.embeddings = None
        if os.path.exists(self.embeddings_path):
            try:
                emb = torch.load(self.embeddings_path, map_location=self.device)
                self.embeddings = emb.to(self.device) if hasattr(emb, "to") else emb
                logger.info("Loaded saved embeddings from %s", self.embeddings_path)
            except Exception:
                logger.exception("Failed to load saved embeddings; re-encoding")
                self._encode_and_cache()
        else:
            self._encode_and_cache()

        self.cross_encoder = None
        if cross_encoder_model and CROSS_AVAILABLE:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model, device=self.device)
                logger.info("Loaded cross-encoder: %s", cross_encoder_model)
            except Exception:
                logger.exception("Failed to load cross-encoder; continuing without it.")
                self.cross_encoder = None

        # tuning
        self.SCORE_SCALE = 3.0
        self.DICT_BOOST_PER = 0.09
        self.MAX_DICT = 1.2
        self.SIGMOID_CLIP = 5.0

    def _encode_and_cache(self):
        texts = [alnum_lower(s["text"]) for s in self.sections]
        logger.info("Encoding %d section texts ...", len(texts))
        emb = self.embed_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        try:
            torch.save(emb.cpu(), self.embeddings_path)
            logger.info("Saved embeddings to %s", self.embeddings_path)
        except Exception:
            logger.warning("Couldn't persist embeddings (permission/IO).")
        self.embeddings = emb.to(self.device) if hasattr(emb, "to") else emb

    def _dictionary_boost(self, query: str, idx: int) -> float:
        if not self.dictionary:
            return 0.0
        sec = self.sections[idx]["section"]
        kws = self.dictionary.get(sec, []) or []
        q_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", query.lower()))
        score = 0.0
        for kw in kws:
            kw_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", kw.lower()))
            if not kw_tokens:
                continue
            if kw_tokens.issubset(q_tokens):
                score += 1.0
            elif kw_tokens & q_tokens:
                score += 0.35
        return min(score * self.DICT_BOOST_PER, self.MAX_DICT)

    def match(self, query: str, top_k: int = 8, rerank_top: Optional[int] = None) -> List[Dict]:
        if not isinstance(query, str) or not query.strip():
            return []

        q_clean = alnum_lower(query)
        q_emb = self.embed_model.encode(q_clean, convert_to_tensor=True)

        # cosine similarities
        cos_scores = util.pytorch_cos_sim(q_emb, self.embeddings)[0].cpu().numpy()
        raw = cos_scores * self.SCORE_SCALE
        boosts = np.array([self._dictionary_boost(query, i) for i in range(len(self.sections))], dtype=float)
        raw = raw + boosts

        # clamp then sigmoid
        raw = np.clip(raw, -self.SIGMOID_CLIP, self.SIGMOID_CLIP)
        confidences = np.array([sigmoid(x) for x in raw]) * 100.0

        # pick candidates
        pick_count = top_k if rerank_top is None else max(top_k, rerank_top)
        idxs = top_k_indices(raw, pick_count)
        candidates = [{"idx": int(i), "raw": float(raw[i]), "cosine": float(cos_scores[i]), "boost": float(boosts[i]), "confidence": float(confidences[i])} for i in idxs]

        # optional cross-encoder rerank
        if rerank_top and self.cross_encoder:
            rerank_ids = [c["idx"] for c in candidates[:rerank_top]]
            pairs = [[query, self.sections[i]["text"]] for i in rerank_ids]
            try:
                xc_scores = self.cross_encoder.predict(pairs)
                xc = np.array(xc_scores)
                if xc.max() != xc.min():
                    xc_norm = (xc - xc.mean()) / (xc.std() + 1e-9)
                else:
                    xc_norm = xc - xc.mean()
                xc_scaled = np.clip(xc_norm * 1.2, -self.SIGMOID_CLIP, self.SIGMOID_CLIP)
                # replace raw & confidence for reranked items
                for i, rid in enumerate(rerank_ids):
                    for c in candidates:
                        if c["idx"] == rid:
                            c["raw"] = float(xc_scaled[i])
                            c["confidence"] = float(sigmoid(c["raw"]) * 100.0)
                candidates.sort(key=lambda x: -x["raw"])
            except Exception:
                logger.exception("Cross-encoder rerank failed")

        final = candidates[:top_k]
        results = []
        for c in final:
            s = self.sections[c["idx"]]
            results.append({
                "section": s["section"],
                "title": s["title"],
                "description": s["description"],
                "score": round(c["raw"], 6),
                "confidence": round(c["confidence"], 3),
                "meta": {"cosine": round(c["cosine"], 6), "boost": round(c["boost"], 6)}
            })
        return results
