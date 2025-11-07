# bns_api.py
from flask import Flask, request, jsonify, render_template
import numpy as np
import os
from bns_matcher import BNSMatcher

app = Flask(__name__, template_folder="templates", static_folder="static")

MATCHER = None

def init_matcher():
    global MATCHER
    if MATCHER is None:
        print("⚖️ Initializing BNS matcher...")
        MATCHER = BNSMatcher(
            dataset_path="bns_dataset.csv",
            dictionary_path="bns_dictionary.json",
            keywords_csv="bns_keywords.csv",
            embeddings_path="bns_embeddings.pt",
            embed_model_name="sentence-transformers/all-mpnet-base-v2",
            cross_encoder_model=None  # set a cross-encoder model name to enable reranking
        )
    return MATCHER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json() or {}
    q = data.get("query", "")
    top_k = int(data.get("top_k", 6))
    rerank = data.get("rerank_top")
    rerank_top = int(rerank) if rerank else None
    matcher = init_matcher()
    res = matcher.match(q, top_k=top_k, rerank_top=rerank_top)
    # ensure JSON-safe types
    for r in res:
        r["score"] = float(r.get("score") or 0.0)
        r["confidence"] = float(r.get("confidence") or 0.0)
    return jsonify({"query": q, "results": res})

if __name__ == "__main__":
    init_matcher()
    app.run(debug=True, host="0.0.0.0", port=5000)
