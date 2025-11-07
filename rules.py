# rules.py
import re
from typing import Optional, Dict

# Basic deterministic rules. Expand as you curate.
# Given query string lowercased, return a dict {"section": "101", "reason": "..."} if a rule matches, else None.

HUMAN_KILL_TERMS = {"kill", "killed", "murder", "stabbed", "shot", "slit", "strangled", "poisoned"}
ANIMAL_KILL_TERMS = {"dog", "cow", "horse", "buffalo", "animal", "pet"}
NEGLIGENCE_TERMS = {"negligent", "negligence", "accidentally", "by mistake", "rash"}
ATTEMPT_TERMS = {"attempt", "tried to", "attempted to"}
THEFT_TERMS = {"steal", "stole", "rob", "robbed", "snatch", "snatched", "theft", "pickpocket"}

def rule_match(query: str) -> Optional[Dict]:
    q = query.lower()
    tokens = set(re.findall(r"\b[a-z0-9']+\b", q))

    # explicit murder/kill of human -> high confidence murder / culpable homicide vs attempt
    if any(t in q for t in HUMAN_KILL_TERMS):
        # if mentions attempt words -> attempt to murder
        if any(a in q for a in ATTEMPT_TERMS):
            return {"section": None, "label": "attempt", "priority": 100, "reason": "attempt detected"}
        # if negligence -> likely culpable homicide by negligence
        if any(n in q for n in NEGLIGENCE_TERMS):
            return {"section": None, "label": "negligence_homicide", "priority": 90, "reason": "negligence words"}
        # if mentions animal -> different
        if any(a in tokens for a in ANIMAL_KILL_TERMS):
            return {"section": None, "label": "animal_killing", "priority": 95, "reason": "animal referenced"}
        return {"section": None, "label": "killing_human", "priority": 100, "reason": "human killing"}

    # theft-related explicit
    if any(t in tokens for t in THEFT_TERMS):
        # snatching/robbery detection
        if "snatch" in q or "snatched" in q:
            return {"section": None, "label": "snatching", "priority": 95, "reason": "snatching explicit"}
        if "rob" in q or "robbed" in q:
            return {"section": None, "label": "robbery", "priority": 95, "reason": "robbery explicit"}
        return {"section": None, "label": "theft", "priority": 90, "reason": "theft explicit"}

    return None
