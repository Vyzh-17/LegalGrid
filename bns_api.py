# bns_api.py
from flask import Flask, request, jsonify, render_template
import os
import json
import numpy as np
import pandas as pd

from bns_matcher import BNSMatcher

app = Flask(__name__, template_folder="templates", static_folder="static")

# instantiate matcher once
MATCHER = None

def init_matcher():
    global MATCHER
    if MATCHER is None:
        print("⚖️ Loading BNS matcher...")
        MATCHER = BNSMatcher(
            dataset_path="bns_dataset.csv",
            dictionary_path="bns_dictionary.json",
            embeddings_path="bns_embeddings.pt",
            embed_model_name="sentence-transformers/all-mpnet-base-v2",
            cross_encoder_model=None  # change to cross-encoder model name to enable rerank
        )
    return MATCHER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json() or {}
    q = data.get("query", "")
    top_k = int(data.get("top_k", 6))
    rerank_top = data.get("rerank_top")
    if rerank_top is not None:
        try:
            rerank_top = int(rerank_top)
        except:
            rerank_top = None

    matcher = init_matcher()
    results = matcher.match(q, top_k=top_k, rerank_top=rerank_top)

    # ensure everything JSON-serializable
    for r in results:
        r["score"] = float(r.get("score") or 0.0)
        r["confidence"] = float(r.get("confidence") or 0.0)

    return jsonify({"query": q, "results": results})

if __name__ == "__main__":
    init_matcher()
    app.run(debug=True, host="0.0.0.0", port=5000)
