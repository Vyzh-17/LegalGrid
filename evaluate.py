# evaluate.py
import pandas as pd
from bns_matcher import BNSMatcher

m = BNSMatcher()
df = pd.read_csv("test_queries.csv", dtype=str).fillna("")
top1 = 0
top3 = 0
n = len(df)
for _, r in df.iterrows():
    q = r["query"]
    correct = str(r["correct_section"]).strip()
    res = m.match(q, top_k=5)
    got = [str(x["section"]).strip() for x in res]
    if correct in got[:1]:
        top1 += 1
    if correct in got[:3]:
        top3 += 1
print("N:", n, "Top-1 acc:", top1/n, "Top-3 acc:", top3/n)
