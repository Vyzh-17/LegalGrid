# build_dictionary.py
"""
Build a curated dictionary for BNS sections:
 - extracts candidate keywords from section titles + descriptions
 - expands with WordNet synonyms (optional)
 - outputs: bns_dictionary.json and bns_keywords.csv (curator-friendly)
"""
import json
import re
import csv
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn

# Try to load spaCy for better keywords (optional)
USE_SPACY = False
try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # user may not have downloaded model
        # set flag but don't fail; will use simple tokenization
        nlp = None
    if nlp:
        USE_SPACY = True
except Exception:
    nlp = None
    USE_SPACY = False

CSV_IN = "bns_dataset.csv"
OUT_JSON = "bns_dictionary.json"
OUT_CSV = "bns_keywords.csv"


def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_tokens(s: str):
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    toks = [t for t in s.split() if len(t) > 2]
    return toks


def extract_candidates_spacy(text: str):
    if not nlp:
        return []
    doc = nlp(text)
    cand = set()
    # nouns & noun_chunks
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2:
            cand.add(chunk.text.lower().strip())
    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN") and len(tok.text) > 2:
            cand.add(tok.lemma_.lower())
    return list(cand)


def synonyms(word: str):
    out = set()
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            w = l.name().replace("_", " ").lower()
            if len(w) > 2:
                out.add(w)
    return out


def build_dictionary(df: pd.DataFrame, expand_synonyms=True):
    mapping = defaultdict(set)
    for _, row in df.iterrows():
        # read many possible column names
        sec = clean_text(row.get("Section") or row.get("section") or row.get("Section ") or row.get("Chapter", ""))
        title = clean_text(row.get("Section _name") or row.get("Section_name") or row.get("Section name") or row.get("title") or "")
        desc = clean_text(row.get("Description") or row.get("description") or "")

        text = f"{title} {desc}".strip()
        candidates = set()

        if USE_SPACY:
            candidates.update(extract_candidates_spacy(text))

        # always add simple tokens too
        candidates.update(simple_tokens(text))

        # filter trivial tokens
        candidates = {c for c in candidates if len(c) > 2 and not c.isdigit()}

        # optionally expand via wordnet (only single words)
        final_keywords = set()
        for c in candidates:
            final_keywords.add(c)
            if expand_synonyms and " " not in c:
                for s in synonyms(c):
                    if len(s) > 2:
                        final_keywords.add(s)

        mapping[sec].update(final_keywords)

    # convert to sorted lists
    out = {sec: sorted(list(sorted(mapping[sec]))) for sec in mapping}
    return out


def save_outputs(mapping, json_path=OUT_JSON, csv_path=OUT_CSV):
    Path(json_path).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    # csv: section,keyword,source (auto). Source is 'auto'
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "keyword", "source"])
        for sec, kws in mapping.items():
            for k in kws:
                writer.writerow([sec, k, "auto"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="csv_in", default=CSV_IN)
    parser.add_argument("--out-json", dest="out_json", default=OUT_JSON)
    parser.add_argument("--out-csv", dest="out_csv", default=OUT_CSV)
    parser.add_argument("--no-syn", dest="no_syn", action="store_true", help="disable synonym expansion")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_in, dtype=str).fillna("")
    print(f"Read dataset with {len(df)} rows")
    mapping = build_dictionary(df, expand_synonyms=not args.no_syn)
    save_outputs(mapping, json_path=args.out_json, csv_path=args.out_csv)
    print(f"Saved dictionary to {args.out_json} and {args.out_csv}")
    print("Open the CSV to curate keywords (remove irrelevant words, add legal phrases).")


if __name__ == "__main__":
    main()
