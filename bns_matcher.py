# bns_matcher.py
import os, json, re, logging
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

# try imports
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

from rules import rule_match

logger = logging.getLogger("BNSMatcher")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

def alnum_lower(s):
    return re.sub(r"[^a-z0-9\s]", " ", str(s).lower())

def sigmoid(x):
    import math
    return 1/(1+math.exp(-x)) if x > -700 else 0.0

class BNSMatcher:
    def __init__(self,
                 dataset_path="bns_dataset.csv",
                 dictionary_path="bns_dictionary.json",
                 keywords_csv="bns_keywords.csv",
                 embeddings_path="bns_embeddings.pt",
                 embed_model_name="sentence-transformers/all-mpnet-base-v2",
                 cross_encoder_model: Optional[str]=None,
                 device: Optional[str]=None):
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(dataset_path)
        self.df = pd.read_csv(dataset_path, dtype=str).fillna("")
        self.sections = []
        for _, row in self.df.iterrows():
            sec = (row.get("Section") or row.get("section") or row.get("Chapter") or "").strip()
            title = (row.get("Section _name") or row.get("Section_name") or row.get("title") or "").strip()
            desc = (row.get("Description") or row.get("description") or "").strip()
            text = f"{title}\n\n{desc}".strip()
            self.sections.append({"section": sec, "title": title, "description": desc, "text": text})

        # dictionary / keywords
        self.dictionary = {}
        if os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, "r", encoding="utf-8") as f:
                    self.dictionary = json.load(f)
            except Exception:
                logger.exception("failed to load dictionary json")

        # load curator CSV if present to override
        if os.path.exists(keywords_csv):
            try:
                km = pd.read_csv(keywords_csv, dtype=str).fillna("")
                for _, r in km.iterrows():
                    sec = str(r["section"]).strip()
                    kw = str(r["keyword"]).strip().lower()
                    if sec not in self.dictionary:
                        self.dictionary[sec] = []
                    if kw not in self.dictionary[sec]:
                        self.dictionary[sec].append(kw)
                logger.info("Loaded keywords CSV - %d entries", len(km))
            except Exception:
                logger.exception("failed to load keywords csv")

        # device
        if device is None:
            device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self.device = device
        logger.info("Using device: %s", self.device)

        if not SBER_AVAILABLE:
            raise RuntimeError("Please install sentence-transformers")

        logger.info("Loading embedder model...")
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)

        self.embeddings_path = embeddings_path
        self.embeddings = None
        if os.path.exists(self.embeddings_path):
            try:
                emb = torch.load(self.embeddings_path, map_location=self.device)
                self.embeddings = emb.to(self.device) if hasattr(emb, "to") else emb
                logger.info("Loaded embeddings from %s", self.embeddings_path)
            except Exception:
                logger.exception("failed to load embeddings; will re-encode")
                self._encode_and_cache()
        else:
            self._encode_and_cache()

        self.cross = None
        if cross_encoder_model and CROSS_AVAILABLE:
            try:
                self.cross = CrossEncoder(cross_encoder_model, device=self.device)
                logger.info("Loaded cross-encoder: %s", cross_encoder_model)
            except Exception:
                logger.exception("Failed to load cross-encoder")

        # parameters
        self.SCORE_SCALE = 4.0
        self.DICT_BOOST = 0.12
        self.MAX_DICT = 1.5
        self.SIG_CLIP = 6.0

    def _encode_and_cache(self):
        texts = [alnum_lower(s["text"]) for s in self.sections]
        logger.info("Encoding %d texts for embeddings...", len(texts))
        emb = self.embed_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        try:
            torch.save(emb.cpu(), self.embeddings_path)
            logger.info("Saved embeddings")
        except Exception:
            logger.warning("Couldn't save embeddings")
        self.embeddings = emb.to(self.device) if hasattr(emb, "to") else emb

    def _dict_boost(self, query, idx):
        if not self.dictionary:
            return 0.0
        sec = self.sections[idx]["section"]
        kws = self.dictionary.get(sec, []) or []
        if not kws:
            return 0.0
        q_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", query.lower()))
        score = 0.0
        for kw in kws:
            kw_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", kw.lower()))
            if not kw_tokens:
                continue
            if kw_tokens.issubset(q_tokens):
                score += 1.2
            elif kw_tokens & q_tokens:
                score += 0.45
        return min(score * self.DICT_BOOST, self.MAX_DICT)

    def match(self, query: str, top_k=8, rerank_top: Optional[int]=None):
        query = str(query or "").strip()
        if not query:
            return []

        # 1. rules layer: if a deterministic rule matches, prefer those labels by boosting relevant sections
        rule = rule_match(query)
        rule_label = rule["label"] if rule else None

        q_clean = alnum_lower(query)
        q_emb = self.embed_model.encode(q_clean, convert_to_tensor=True)
        cos = util.pytorch_cos_sim(q_emb, self.embeddings)[0].cpu().numpy()
        raw = cos * self.SCORE_SCALE

        boosts = [self._dict_boost(query, i) for i in range(len(self.sections))]

        # rule-based section boost (map labels to heuristic)
        if rule_label:
            for i, s in enumerate(self.sections):
                text = (s["text"] or "").lower()
                sec = s["section"]
                # apply strong heuristics
                if rule_label == "killing_human":
                    # boost sections whose descriptions contain murder/culpable/homicide
                    if re.search(r"\bmurder\b|\bculpable homicide\b|\bkill\b", text):
                        boosts[i] += 1.4
                elif rule_label == "animal_killing":
                    if re.search(r"\banimal\b|\bdog\b|\bkill\b|\bkilled\b", text):
                        boosts[i] += 1.2
                elif rule_label == "theft":
                    if re.search(r"\btheft\b|\bstolen\b|\brobbery\b|\bsnatch\b|\bmisappropriation\b", text):
                        boosts[i] += 1.3
                elif rule_label == "snatching":
                    if re.search(r"\bsnatch\b|\bsnatching\b", text):
                        boosts[i] += 1.8
                elif rule_label == "robbery":
                    if re.search(r"\brobbery\b|\brobber\b", text):
                        boosts[i] += 1.8

        raw = raw + np.array(boosts)
        raw = np.clip(raw, -self.SIG_CLIP, self.SIG_CLIP)
        confidences = [float(sigmoid(v) * 100.0) for v in raw]

        # choose candidates
        pick = max(top_k, rerank_top or top_k)
        idxs = list(np.argsort(-raw)[:pick])
        candidates = [{"idx": int(i), "raw": float(raw[i]), "cos": float(cos[i]), "boost": float(boosts[i]), "conf": float(confidences[i])} for i in idxs]

        # optional cross-encoder rerank
        if rerank_top and self.cross:
            rerank_ids = [c["idx"] for c in candidates[:rerank_top]]
            pairs = [[query, self.sections[i]["text"]] for i in rerank_ids]
            try:
                xc = self.cross.predict(pairs)
                xc = np.array(xc)
                xc_norm = (xc - xc.mean()) / (xc.std() + 1e-9) if xc.std() > 0 else (xc - xc.mean())
                xc_scaled = np.clip(xc_norm * 1.5, -self.SIG_CLIP, self.SIG_CLIP)
                for i, rid in enumerate(rerank_ids):
                    for c in candidates:
                        if c["idx"] == rid:
                            c["raw"] = float(xc_scaled[i])
                            c["conf"] = float(sigmoid(c["raw"]) * 100.0)
                candidates.sort(key=lambda x: -x["raw"])
            except Exception:
                logger.exception("cross-encoder failed")

        results = []
        for c in candidates[:top_k]:
            s = self.sections[c["idx"]]
            results.append({
                "section": s["section"],
                "title": s["title"],
                "description": s["description"],
                "score": round(c["raw"], 6),
                "confidence": round(c["conf"], 3),
                "meta": {"cosine": round(c["cos"], 6), "boost": round(c["boost"], 6)}
            })
        return results
