# build_dictionary.py
import json, re, csv, argparse
from collections import defaultdict
import pandas as pd
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn

CSV_IN = "bns_dataset.csv"
OUT_JSON = "bns_dictionary.json"
OUT_CSV = "bns_keywords.csv"

def clean(s):
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s).replace("\r"," ").replace("\n"," ")).strip()

def simple_tokens(s):
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower())
    return [t for t in s.split() if len(t) > 2]

def synonyms(w):
    out = set()
    for syn in wn.synsets(w):
        for l in syn.lemmas():
            out.add(l.name().replace("_", " ").lower())
    return out

def build(df, expand_syn=True):
    mapping = defaultdict(set)
    for _, row in df.iterrows():
        sec = clean(row.get("Section") or row.get("section") or row.get("Chapter") or "")
        title = clean(row.get("Section _name") or row.get("Section_name") or row.get("title") or "")
        desc = clean(row.get("Description") or row.get("description") or "")
        text = f"{title} {desc}"
        toks = set(simple_tokens(text))
        for t in list(toks):
            mapping[sec].add(t)
            if expand_syn and " " not in t:
                for s in synonyms(t):
                    if len(s) > 2:
                        mapping[sec].add(s)
    # write json and csv
    out = {k: sorted(list(v)) for k,v in mapping.items()}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section","keyword","source"])
        for sec,kws in out.items():
            for kw in kws:
                w.writerow([sec,kw,"auto"])
    print("Saved", OUT_JSON, "and", OUT_CSV)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="csv", default=CSV_IN)
    p.add_argument("--no-syn", dest="no_syn", action="store_true")
    args = p.parse_args()
    df = pd.read_csv(args.csv, dtype=str).fillna("")
    build(df, expand_syn=not args.no_syn)
