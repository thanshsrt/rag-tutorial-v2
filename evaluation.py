import json
import re
import os
from datetime import datetime

def evaluate_review(diff_text: str, review_text: str, rag_sources: list, rag_text: str = "") -> dict:
    """
    Heuristic evaluation for 3B model code reviews.
    Catches: hallucinations, generic filler, lazy outputs, domain mismatches,
             silent mode switches, and workflow irrelevance.
    """
    score = 10.0
    metrics = {
        "has_citations": False,
        "diff_overlap": 0,
        "filler_count": 0,
        "hallucinated_diff": False,
        "generic_title_reuse": False,
        "uses_code_formatting": False,
        "word_count": 0,
        "rag_empty": False,
        "domain_mismatch": False,
        "specificity_score": 0,
        "silent_diff_only": False,
        "no_indexing_suggestion": False
    }
    
    # === RAG EMPTY DETECTION ===
    rag_is_empty = (
        not rag_sources 
        or len(rag_sources) == 0 
        or "NOTICE:" in rag_text 
        or "No matching internal patterns" in rag_text
    )
    metrics["rag_empty"] = rag_is_empty
    
    # === DOMAIN MISMATCH (claims internal comparison when none exists) ===
    if rag_is_empty:
        internal_comparison_phrases = [
            "internal codebase", "our codebase", "existing code",
            "compared to", "aligns with", "matches our", "in our",
            "consistent with", "follows our", "violates our"
        ]
        claims_internal = any(p in review_text.lower() for p in internal_comparison_phrases)
        
        cited_files = set(re.findall(r'`([^\n`]+\.(?:py|js|ts|tsx|jsx|go|rs|java))`', review_text))
        diff_files = set(re.findall(r'(?:\+\+\+) [ab]/(.+?)(?:\t|\n)', diff_text))
        diff_basenames = {os.path.basename(f) for f in diff_files}
        suspicious_citations = cited_files - diff_basenames
        
        if claims_internal or len(suspicious_citations) > 0:
            metrics["domain_mismatch"] = True
            score -= 5.0
    
    # === SILENT DIFF-ONLY MODE (doesn't admit RAG is empty) ===
    if rag_is_empty:
        honesty_phrases = [
            "no matching internal", "no internal patterns", "review based on diff",
            "diff only", "no relevant code found", "based solely on", "internal context"
        ]
        admits_empty_rag = any(p in review_text.lower() for p in honesty_phrases)
        if not admits_empty_rag:
            score -= 3.0
            metrics["silent_diff_only"] = True
    
    # === WORKFLOW RELEVANCE (cross-domain PRs should suggest indexing) ===
    if rag_is_empty:
        suggests_indexing = any(p in review_text.lower() for p in [
            "add to index", "index this", "include in codebase", "similar pattern in"
        ])
        if not suggests_indexing:
            metrics["no_indexing_suggestion"] = True
            score -= 1.0
    
    # === CITATION CHECK (skip if RAG empty) ===
    if not rag_is_empty and rag_sources:
        short_names = [os.path.basename(src) for src in rag_sources if isinstance(src, str)]
        found = [name for name in short_names if name.lower() in review_text.lower()]
        metrics["has_citations"] = len(found) > 0
        if not metrics["has_citations"]:
            score -= 2.0
    
    # === DIFF OVERLAP ===
    diff_methods = set(re.findall(r'(?:def|class|function|const)\s+(\w+)', diff_text))
    review_methods = set(re.findall(r'`(\w+)`', review_text))
    metrics["diff_overlap"] = len(review_methods.intersection(diff_methods))
    if metrics["diff_overlap"] == 0 and len(diff_methods) > 0:
        score -= 3.0
    
    # === SPECIFICITY CHECK ===
    specificity_markers = [
        "line", "line ", "line:", "lines ", "l",
        "replace", "change", "remove", "add ",
        "with `", "to `", "from `"
    ]
    metrics["specificity_score"] = sum(1 for m in specificity_markers if m in review_text.lower())
    if metrics["specificity_score"] < 2:
        score -= 2.5
    
    # === FILLER PENALTY ===
    fillers = [
        "it is important to", "ensure that", "properly", "consider",
        "add tests", "improve docs", "best practice", "should be",
        "need to", "must be", "recommended to", "handle potential",
        "improve the handling", "better", "more robust",
        "unclear if", "necessary to ensure", "thorough review"
    ]
    metrics["filler_count"] = sum(1 for f in fillers if f in review_text.lower())
    score -= metrics["filler_count"] * 0.75
    
    # === HALLUCINATED DIFF BLOCKS ===
    if "```diff" in review_text:
        if "+ " not in review_text and "- " not in review_text:
            metrics["hallucinated_diff"] = True
            score -= 2.5
    
    # === TITLE REUSE ===
    titles = re.findall(r'(?:TITLE|Title|title):\s*([^\n|]+)', review_text)
    if len(titles) >= 2 and len(set(t.strip().lower() for t in titles)) == 1:
        metrics["generic_title_reuse"] = True
        score -= 1.5
    
    # === CODE FORMATTING ===
    metrics["uses_code_formatting"] = "`" in review_text
    if not metrics["uses_code_formatting"]:
        score -= 1.0
    
    # === LENGTH CHECK ===
    metrics["word_count"] = len(review_text.split())
    if metrics["word_count"] < 80:
        score -= 1.5
    
    final_score = max(0.0, round(score, 1))
    
    # Dynamic pass threshold
    pass_threshold = 8.5 if rag_is_empty else 7.0
    
    return {
        "timestamp": datetime.now().isoformat(),
        "score": final_score,
        "passed": final_score >= pass_threshold,
        "pass_threshold": pass_threshold,
        "metrics": metrics,
        "review_snippet": review_text[:120] + "..." if len(review_text) > 120 else review_text
    }

def log_evaluation(result: dict):
    with open("review_evaluations.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")