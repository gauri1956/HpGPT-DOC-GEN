"""
data_fusion_agent.py  -  HPGPT Cross-File Intelligence Layer
=============================================================
Improvements implemented
------------------------
1.  Reuses entities already extracted by data_agent -- no duplicate processing.
2.  Entity synonym handling: dept/division/business unit/function -> "department".
3.  Confidence-scored relationship discovery -- weak links filtered out.
4.  Cross-file correlation: numerical bridging across semantic links.
5.  Insight prioritisation: Critical / High / Medium / Low labels.
6.  File relevance scoring: ranks files against the user prompt.
7.  Full integration surface for agent_orchestrator.py.
 
Public API
----------
    from data_fusion_agent import DataFusionAgent
 
    agent  = DataFusionAgent(file_data, user_prompt)
    result = agent.run()
 
    result.cross_file_context  -> str  (prepended to LLM prompt)
    result.ranked_files        -> list[str]  (most-relevant first)
    result.links               -> list[LinkRecord]
    result.insights            -> list[InsightRecord]
    result.summary_log         -> str  (human-readable debug summary)
"""
 
from __future__ import annotations
 
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any
 
import pandas as pd
 
logger = logging.getLogger(__name__)
 
 
# ==============================================================================
# 1.  SYNONYM TABLE  -  maps raw column fragments -> canonical entity name
# ==============================================================================
 
_SYNONYMS: dict[str, list[str]] = {
    "department":   ["department", "dept", "division", "business unit",
                     "function", "vertical", "bu", "biz unit"],
    "region":       ["region", "zone", "area", "geography", "territory",
                     "cluster", "circle"],
    "location":     ["location", "city", "state", "branch", "office",
                     "plant", "site", "depot", "terminal", "hub"],
    "product":      ["product", "item", "sku", "material", "commodity",
                     "fuel", "grade", "offering", "service"],
    "employee":     ["employee", "emp", "staff", "worker", "personnel",
                     "headcount", "associate", "executive"],
    "vendor":       ["vendor", "supplier", "contractor", "agency",
                     "partner", "subcontractor"],
    "customer":     ["customer", "client", "account", "buyer",
                     "consumer", "dealer", "distributor"],
    "category":     ["category", "type", "class", "segment", "group",
                     "tier", "classification"],
    "project":      ["project", "initiative", "program", "scheme",
                     "work order", "assignment"],
    "period":       ["month", "quarter", "year", "period", "fy",
                     "date", "week", "fiscal", "timeline"],
    "cost_center":  ["cost center", "cost centre", "profit center",
                     "cost code", "gl code"],
    "asset":        ["asset", "equipment", "machine", "facility",
                     "tank", "vehicle", "infrastructure"],
    "team":         ["team", "unit", "squad", "group", "cluster",
                     "pod", "crew"],
    "role":         ["role", "designation", "title", "position",
                     "grade", "level", "band"],
}
 
# Reverse lookup: keyword fragment -> canonical name
_KW_TO_ENTITY: dict[str, str] = {
    kw: entity
    for entity, kws in _SYNONYMS.items()
    for kw in kws
}
 
 
def canonical_entity(col_name: str) -> str | None:
    """
    Map a raw column name to its canonical entity type using synonym table.
    Returns None if no match found.
    """
    cl = col_name.lower().strip()
    # Longest-match first to avoid "group" matching before "cost_center"
    for kw in sorted(_KW_TO_ENTITY, key=len, reverse=True):
        if kw in cl:
            return _KW_TO_ENTITY[kw]
    return None


_METRIC_SYNONYMS: dict[str, list[str]] = {
    "attrition_rate": ["attrition rate", "attrition %", "resignation %", "turnover rate", "employee turnover", "exit rate", "attrition"],
    "revenue": ["revenue", "sales", "turnover", "income", "billing", "receipts", "sales volume", "petrol sales", "diesel sales", "lpg sales"],
    "headcount": ["headcount", "employee count", "staff count", "active employees", "total employees", "no of employees", "strength"],
    "salary": ["salary", "compensation", "pay", "wages", "package", "cost to company", "ctc"],
    "inventory": ["inventory", "stock", "closing stock", "opening stock", "quantity", "stock quantity"],
    "profit": ["profit", "margin", "earnings", "net income", "p&l", "profitability"],
}


def canonical_metric(col_name: str) -> str:
    cl = col_name.lower().strip()
    for metric, keywords in _METRIC_SYNONYMS.items():
        if any(kw in cl for kw in keywords):
            return metric
    return re.sub(r'[^a-z0-9]', '', cl)


def detect_column_unit(col_name: str) -> str:
    cl = col_name.lower()
    if any(u in cl for u in ["kl", "kiloliter"]):
        return "volume_kl"
    if any(u in cl for u in ["mt", "tonne", "ton"]):
        return "volume_mt"
    if any(u in cl for u in ["inr", "lakh", "rs", "rupee", "usd", "$", "wages", "salary", "compensation"]):
        return "currency"
    if any(u in cl for u in ["%", "pct", "percent", "rate", "ratio"]):
        return "percentage"
    if any(u in cl for u in ["employee", "staff", "headcount", "count", "no of"]):
        return "count"
    if any(u in cl for u in ["hours", "hrs"]):
        return "hours"
    return "generic_numeric"
 
 
# ==============================================================================
# 2.  DATA STRUCTURES
# ==============================================================================
 
@dataclass
class LinkRecord:
    file_a:        str
    col_a:         str
    file_b:        str
    col_b:         str
    entity_type:   str
    shared_values: list[str]
    confidence:    float          # 0.0 - 1.0
    value_overlap: float          # fraction of values shared
 
 
@dataclass
class InsightRecord:
    title:       str
    detail:      str
    priority:    str              # "Critical" | "High" | "Medium" | "Low"
    files:       list[str]
    entity_type: str
    score:       float            # raw priority score
 
 
# ==============================================================================
# 3.  FILE RELEVANCE SCORING
# ==============================================================================
 
_RELEVANCE_SIGNALS: dict[str, list[str]] = {
    "financial":   ["revenue", "profit", "cost", "expense", "budget",
                    "p&l", "finance", "margin", "loss", "turnover"],
    "hr":          ["employee", "headcount", "attrition", "salary",
                    "recruitment", "training", "hr", "workforce"],
    "sales":       ["sales", "target", "achievement", "volume",
                    "customer", "dealer", "order", "pipeline"],
    "operations":  ["inventory", "stock", "dispatch", "receipt",
                    "production", "capacity", "throughput", "plant"],
    "performance": ["kpi", "performance", "target", "actual",
                    "variance", "score", "rating", "benchmark"],
    "risk":        ["risk", "compliance", "audit", "incident",
                    "safety", "non-compliance", "violation"],
}
 
 
def score_file_relevance(file_data: dict, user_prompt: str) -> list[tuple[str, float]]:
    """
    Score each file 0-1 against the user prompt.
    Returns list of (filename, score) sorted descending.
    """
    prompt_lower = user_prompt.lower()
 
    # Identify which domains the prompt touches
    prompt_domains: set[str] = set()
    for domain, kws in _RELEVANCE_SIGNALS.items():
        if any(kw in prompt_lower for kw in kws):
            prompt_domains.add(domain)
 
    scores: dict[str, float] = {}
 
    for fname, data in file_data.items():
        # Gather all column text + file name
        cols: list[str] = []
        if data.get("type") == "multi_sheet":
            for s in data.get("sheets", {}).values():
                cols.extend(s.get("columns", []))
        else:
            cols = data.get("columns", [])
 
        col_text = " ".join(cols).lower() + " " + fname.lower()
 
        file_score = 0.0
 
        # If prompt has domain signals, score by domain overlap
        if prompt_domains:
            for domain in prompt_domains:
                kws = _RELEVANCE_SIGNALS[domain]
                hits = sum(1 for kw in kws if kw in col_text)
                if hits:
                    file_score += hits * 0.15
 
        # Direct keyword match between prompt words and column names
        prompt_words = set(re.findall(r'\b\w{4,}\b', prompt_lower))
        col_words    = set(re.findall(r'\b\w{4,}\b', col_text))
        direct_hits  = len(prompt_words & col_words)
        file_score  += direct_hits * 0.1
 
        # Intent bonus: analytical files score higher for report prompts
        intent = data.get("intent", "informational")
        if "report" in prompt_lower or "analysis" in prompt_lower:
            if intent == "analytical":
                file_score += 0.2
 
        scores[fname] = round(min(file_score, 1.0), 3)
 
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"[DataFusion] File relevance: {ranked}")
    return ranked
 
 
# ==============================================================================
# 4.  ENTITY MAP BUILDER  -  reuses data_agent extracted entities
# ==============================================================================
 
def _build_entity_map(file_data: dict) -> dict[str, dict[str, dict]]:
    """
    Reuse entities already extracted by data_agent._summarize_df().
    Falls back to column-name scanning if entity key absent.
 
    Returns:
        {filename: {entity_type: {columns: [...], sample_values: [...]}}}
    """
    result: dict[str, dict] = {}
 
    for fname, data in file_data.items():
        # -- Try reusing data_agent entities first -------------------------
        if data.get("type") == "multi_sheet":
            # Multi-sheet: aggregate from cross_sheet_entities if present,
            # else merge per-sheet entity dicts
            if "entities" in data:
                result[fname] = data["entities"]
                continue
            merged: dict[str, dict] = {}
            for sdata in data.get("sheets", {}).values():
                for etype, info in sdata.get("entities", {}).items():
                    if etype not in merged:
                        merged[etype] = {"columns": [], "sample_values": []}
                    for col in info.get("columns", []):
                        if col not in merged[etype]["columns"]:
                            merged[etype]["columns"].append(col)
                    existing = set(merged[etype]["sample_values"])
                    merged[etype]["sample_values"] = list(
                        existing | set(info.get("sample_values", []))
                    )[:20]
            result[fname] = merged
            continue
 
        # Single-frame tabular or text
        if "entities" in data and data["entities"]:
            result[fname] = data["entities"]
            continue
 
        # -- Fallback: scan columns ourselves -----------------------------
        entities: dict[str, dict] = {}
        for col in data.get("columns", []):
            etype = canonical_entity(col)
            if etype is None:
                continue
            if etype not in entities:
                entities[etype] = {"columns": [], "sample_values": []}
            if col not in entities[etype]["columns"]:
                entities[etype]["columns"].append(col)
            # Pull values from sample rows
            for row in data.get("sample", []):
                v = str(row.get(col, "")).strip()
                if v and v not in ("nan", "None", ""):
                    if v not in entities[etype]["sample_values"]:
                        entities[etype]["sample_values"].append(v)
            entities[etype]["sample_values"] = entities[etype]["sample_values"][:20]
 
        result[fname] = entities
 
    return result
 
 
# ==============================================================================
# 5.  LINK DISCOVERY WITH CONFIDENCE SCORING
# ==============================================================================
 
_MIN_LINK_CONFIDENCE = 0.35   # links below this are discarded
 
 
def _discover_links(entity_map: dict[str, dict]) -> list[LinkRecord]:
    """
    Find cross-file links with confidence scores.
 
    Confidence formula:
        base          = 0.5   (same canonical entity type found in both files)
        value_overlap += 0.4  (fraction of shared distinct values, capped at 0.4)
        col_name_sim  += 0.1  (if raw column names are similar, not just same type)
    """
    links: list[LinkRecord] = []
 
    for fa, fb in combinations(entity_map.keys(), 2):
        ents_a = entity_map[fa]
        ents_b = entity_map[fb]
        shared_types = set(ents_a.keys()) & set(ents_b.keys())
 
        for etype in shared_types:
            info_a = ents_a[etype]
            info_b = ents_b[etype]
 
            vals_a = set(v.lower() for v in info_a.get("sample_values", []))
            vals_b = set(v.lower() for v in info_b.get("sample_values", []))
 
            union      = vals_a | vals_b
            intersect  = vals_a & vals_b
            overlap    = len(intersect) / len(union) if union else 0.0
 
            # Period columns don't need value overlap (date ranges may differ)
            if etype == "period":
                overlap = 0.5
 
            confidence = 0.5 + (overlap * 0.4)
 
            # Small bonus if raw column names are similar
            col_a = info_a["columns"][0] if info_a["columns"] else ""
            col_b = info_b["columns"][0] if info_b["columns"] else ""
            if col_a.lower() == col_b.lower():
                confidence += 0.1
            confidence = round(min(confidence, 1.0), 3)
 
            if confidence < _MIN_LINK_CONFIDENCE:
                logger.debug(
                    f"[DataFusion] Discarded low-confidence link "
                    f"{fa}.{col_a}  {fb}.{col_b} "
                    f"entity={etype} confidence={confidence}"
                )
                continue
 
            links.append(LinkRecord(
                file_a        = fa,
                col_a         = col_a,
                file_b        = fb,
                col_b         = col_b,
                entity_type   = etype,
                shared_values = sorted(intersect)[:10],
                confidence    = confidence,
                value_overlap = round(overlap, 3),
            ))
            logger.info(
                f"[DataFusion] Link: {fa}.{col_a}  {fb}.{col_b} "
                f"entity={etype} confidence={confidence} overlap={overlap:.2f}"
            )
 
    # Sort by confidence descending
    links.sort(key=lambda l: l.confidence, reverse=True)
    return links
 
 
# ==============================================================================
# 6.  NUMERIC FUSION  -  cross-file aggregates per shared entity value
# ==============================================================================
 
def _fuse_numerics(
    file_data: dict,
    links: list[LinkRecord]
) -> list[dict]:
    """
    For each high-confidence link, use the aggregated entity_stats computed from
    the full datasets to compare metrics across files. Returns list of fused records.
    """
    fused: list[dict] = []
    
    def get_file_entity_stats(fname: str) -> list[dict]:
        data = file_data[fname]
        if data.get("type") == "multi_sheet":
            all_stats = []
            for sheet_name, sdata in data.get("sheets", {}).items():
                if sdata.get("entity_stats"):
                    for est in sdata["entity_stats"]:
                        est_copy = est.copy()
                        est_copy["sheet"] = sheet_name
                        all_stats.append(est_copy)
            return all_stats
        return data.get("entity_stats", [])

    for link in links:
        if link.confidence < 0.5:
            continue
        
        stats_a = get_file_entity_stats(link.file_a)
        stats_b = get_file_entity_stats(link.file_b)
        
        stats_a_filtered = [s for s in stats_a if s["entity_col"] == link.col_a]
        stats_b_filtered = [s for s in stats_b if s["entity_col"] == link.col_b]
        
        fa_tag = link.file_a.split(".")[0][:8]
        fb_tag = link.file_b.split(".")[0][:8]
        
        for sa in stats_a_filtered:
            ncol_a = sa["numeric_col"]
            metric_a = canonical_metric(ncol_a)
            unit_a = detect_column_unit(ncol_a)
            
            for sb in stats_b_filtered:
                ncol_b = sb["numeric_col"]
                metric_b = canonical_metric(ncol_b)
                unit_b = detect_column_unit(ncol_b)
                
                is_match = (metric_a == metric_b and unit_a == unit_b)
                is_relation = (metric_a != metric_b)
                
                if is_match or is_relation:
                    shared_vals = set(sa["stats"].keys()) & set(sb["stats"].keys())
                    for val in shared_vals:
                        val_stats_a = sa["stats"][val]
                        val_stats_b = sb["stats"][val]
                        
                        num_summary = {
                            f"{ncol_a}_in_{fa_tag}": val_stats_a,
                            f"{ncol_b}_in_{fb_tag}": val_stats_b,
                        }
                        
                        fused.append({
                            "entity_type":     link.entity_type,
                            "entity_value":    val,
                            "source_files":    [link.file_a, link.file_b],
                            "join_cols":       [link.col_a, link.col_b],
                            "numeric_summary": num_summary,
                            "link_confidence": link.confidence,
                            "is_canonical_match": is_match,
                        })
    return fused
 
 
# ==============================================================================
# 7.  INSIGHT PRIORITISATION
# ==============================================================================
 
def _prioritise_insights(
    links: list[LinkRecord],
    fused: list[dict],
    file_data: dict,
    entity_map: dict,
) -> list[InsightRecord]:
    """
    Generate and rank cross-file insights.
 
    Priority score factors:
        link confidence          (0 - 1.0)
        entity importance weight (department/region rank higher than period)
        numeric variance ratio   (high spread -> more interesting)
        number of shared values  (more overlap -> more actionable)
    """
    _ENTITY_IMPORTANCE = {
        "department": 1.0,  "region":     0.95, "location":   0.9,
        "product":    0.85, "employee":   0.85, "customer":   0.8,
        "vendor":     0.7,  "category":   0.7,  "role":       0.7,
        "project":    0.65, "cost_center":0.65, "asset":      0.6,
        "team":       0.6,  "period":     0.5,
    }
 
    insights: list[InsightRecord] = []
 
    # -- Insight type A: link-level observations -------------------------------
    for link in links:
        entity_weight = _ENTITY_IMPORTANCE.get(link.entity_type, 0.5)
        score = (link.confidence * 0.5
                 + entity_weight * 0.3
                 + min(len(link.shared_values) / 10, 1.0) * 0.2)
 
        if link.shared_values:
            sample_vals = ", ".join(link.shared_values[:4])
            detail = (
                f"{link.file_a} and {link.file_b} share "
                f"{len(link.shared_values)} common {link.entity_type} "
                f"values (e.g. {sample_vals}). "
                f"Link confidence: {link.confidence:.0%}. "
                f"Analyse these shared {link.entity_type}s for correlated trends."
            )
        else:
            detail = (
                f"{link.file_a} and {link.file_b} both contain a "
                f"{link.entity_type} dimension. "
                f"Align reporting on this shared axis."
            )
 
        title = (
            f"Cross-file {link.entity_type.title()} connection: "
            f"{link.file_a}  {link.file_b}"
        )
        insights.append(InsightRecord(
            title       = title,
            detail      = detail,
            priority    = _score_to_priority(score),
            files       = [link.file_a, link.file_b],
            entity_type = link.entity_type,
            score       = round(score, 3),
        ))
 
    # -- Insight type B: numeric fusion observations & contradiction detection -
    for rec in fused:
        if not rec.get("is_canonical_match", True):
            continue
        if not rec["numeric_summary"]:
            continue

        entity_weight = _ENTITY_IMPORTANCE.get(rec["entity_type"], 0.5)
        keys = list(rec["numeric_summary"].keys())
        if len(keys) >= 2:
            key_a, key_b = keys[0], keys[1]
            stats_a = rec["numeric_summary"][key_a]
            stats_b = rec["numeric_summary"][key_b]
            
            mean_a = stats_a.get("mean", 0.0)
            mean_b = stats_b.get("mean", 0.0)
            
            diff = abs(mean_a - mean_b)
            avg_mean = (mean_a + mean_b) / 2.0
            rel_diff = diff / avg_mean if avg_mean != 0 else 0.0
            
            # Determine if values are contradictory or consistent
            if rel_diff > 0.15 and diff > 0.5:
                title = f"Data Contradiction: {rec['entity_type'].title()} '{rec['entity_value']}' metric mismatch"
                detail = (
                    f"Contradiction detected for {rec['entity_type']} '{rec['entity_value']}': "
                    f"'{key_a}' is {mean_a} in {rec['source_files'][0]}, but "
                    f"'{key_b}' is {mean_b} in {rec['source_files'][1]}. "
                    f"This is a variance of {diff:.1f} ({rel_diff:.1%}). "
                    f"Investigate the data sources for reporting discrepancies."
                )
                priority = "Critical" if rel_diff > 0.3 else "High"
                score = 0.5 + rel_diff * 0.5
            else:
                title = f"Data Bridge: {rec['entity_type'].title()} '{rec['entity_value']}' numeric consistency"
                detail = (
                    f"Consistent data found for {rec['entity_type']} '{rec['entity_value']}' across files: "
                    f"'{key_a}' is {mean_a} in {rec['source_files'][0]} and "
                    f"'{key_b}' is {mean_b} in {rec['source_files'][1]}. "
                    f"This confirms alignment on this shared metric."
                )
                priority = "Medium"
                score = 0.4 + (1.0 - rel_diff) * 0.2
                
            insights.append(InsightRecord(
                title       = title,
                detail      = detail,
                priority    = priority,
                files       = rec["source_files"],
                entity_type = rec["entity_type"],
                score       = round(score, 3),
            ))

    # -- Insight type C: Trend signals -----------------------------------------
    for fname, data in file_data.items():
        trends = data.get("trends", {})
        if data.get("type") == "multi_sheet":
            trends = {}
            for sname, sdata in data.get("sheets", {}).items():
                if sdata.get("trends"):
                    trends.update(sdata["trends"])
                    
        for col, t in trends.items():
            if t.get("direction") in ("up", "down"):
                title = f"Trend Signal: {fname} - {col} is trending {t['direction']}"
                detail = (
                    f"In {fname}, metric '{col}' is {t['trend_signal']} "
                    f"over period '{t['period_column']}'."
                )
                insights.append(InsightRecord(
                    title       = title,
                    detail      = detail,
                    priority    = "High" if abs(t["pct_change"]) > 15 else "Medium",
                    files       = [fname],
                    entity_type = "period",
                    score       = round(0.4 + min(abs(t["pct_change"]) / 100, 0.5), 3),
                ))
 
    # -- Insight type D: Cross-file Correlation & Dependency Detection ---------
    # Group fused records by (file_a, file_b, ncol_a, ncol_b, entity_type)
    groups = defaultdict(list)
    for rec in fused:
        keys = list(rec["numeric_summary"].keys())
        if len(keys) < 2:
            continue
        key_a, key_b = keys[0], keys[1]
        
        # Strip suffix from key_a and key_b to get raw column names
        ncol_a = re.sub(r'_in_[a-f0-9]+$', '', key_a)
        ncol_b = re.sub(r'_in_[a-f0-9]+$', '', key_b)
        
        file_a = rec["source_files"][0]
        file_b = rec["source_files"][1]
        
        group_key = (file_a, file_b, ncol_a, ncol_b, rec["entity_type"])
        groups[group_key].append(rec)
        
    for (file_a, file_b, ncol_a, ncol_b, etype), recs in groups.items():
        if len(recs) < 3:
            continue
            
        list_a = []
        list_b = []
        val_names = []
        
        for r in recs:
            keys = list(r["numeric_summary"].keys())
            sa = r["numeric_summary"][keys[0]]
            sb = r["numeric_summary"][keys[1]]
            
            mean_a = sa.get("mean")
            mean_b = sb.get("mean")
            
            if mean_a is not None and mean_b is not None:
                list_a.append(mean_a)
                list_b.append(mean_b)
                val_names.append((r["entity_value"], mean_a, mean_b))
                
        if len(list_a) >= 3:
            s_a = pd.Series(list_a)
            s_b = pd.Series(list_b)
            if s_a.std() > 0 and s_b.std() > 0:
                r_coeff = s_a.corr(s_b)
                if not pd.isna(r_coeff) and abs(r_coeff) >= 0.75:
                    direction_str = "positive" if r_coeff > 0 else "negative"
                    priority = "Critical" if abs(r_coeff) >= 0.90 else "High"
                    
                    samples = []
                    for val, ma, mb in val_names[:3]:
                        samples.append(f"'{val}' ({ncol_a}={ma:.1f}, {ncol_b}={mb:.1f})")
                    sample_str = ", ".join(samples)
                    
                    title = f"Cross-file Correlation: '{ncol_a}' and '{ncol_b}' ({direction_str})"
                    detail = (
                        f"A strong {direction_str} correlation of {r_coeff:.2f} was detected between "
                        f"'{ncol_a}' in {file_a} and '{ncol_b}' in {file_b} across the shared '{etype}' dimension. "
                        f"Sample pairings: {sample_str}."
                    )
                    
                    insights.append(InsightRecord(
                        title       = title,
                        detail      = detail,
                        priority    = priority,
                        files       = [file_a, file_b],
                        entity_type = etype,
                        score       = round(0.5 + abs(r_coeff) * 0.5, 3)
                    ))

    # Sort: Critical -> High -> Medium -> Low, then by score desc
    _ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    insights.sort(key=lambda i: (_ORDER[i.priority], -i.score))
    return insights
 
 
def _score_to_priority(score: float) -> str:
    if score >= 0.75:   return "Critical"
    if score >= 0.55:   return "High"
    if score >= 0.35:   return "Medium"
    return "Low"
 
 
# ==============================================================================
# 8.  CONTEXT RENDERER
# ==============================================================================
 
def _render_context(
    file_data:    dict,
    entity_map:   dict,
    links:        list[LinkRecord],
    fused:        list[dict],
    insights:     list[InsightRecord],
    ranked_files: list[tuple[str, float]],
) -> str:
    lines: list[str] = []
 
    lines.append("===============================================")
    lines.append("  CROSS-FILE INTELLIGENCE CONTEXT (HPGPT)")
    lines.append("===============================================\n")
 
    # File relevance ranking
    lines.append("FILE RELEVANCE RANKING:")
    for fname, rel_score in ranked_files:
        intent = file_data[fname].get("intent", "")
        lines.append(f"  [{rel_score:.2f}] {fname}  (intent: {intent})")
 
    # Entity dimensions per file
    lines.append("\nENTITY DIMENSIONS DETECTED:")
    for fname, entities in entity_map.items():
        if not entities:
            lines.append(f"  {fname}: no business entity columns")
            continue
        lines.append(f"  {fname}:")
        for etype, info in entities.items():
            sv = ", ".join(info.get("sample_values", [])[:5])
            lines.append(
                f"    - {etype}: cols={info['columns']} | "
                f"sample values: {sv}"
            )
 
    # Cross-file links
    if links:
        lines.append(f"\nCROSS-FILE ENTITY LINKS ({len(links)} found, "
                     f"confidence  {_MIN_LINK_CONFIDENCE}):")
        for lnk in links:
            flag = " Critical" if lnk.confidence >= 0.75 else \
                   " High"     if lnk.confidence >= 0.55 else " Medium"
            lines.append(
                f"  {flag}  [{lnk.entity_type.upper()}]  "
                f"{lnk.file_a}.{lnk.col_a}  "
                f"{lnk.file_b}.{lnk.col_b}  "
                f"(confidence={lnk.confidence:.0%}, "
                f"overlap={lnk.value_overlap:.0%})"
            )
            if lnk.shared_values:
                lines.append(
                    f"      shared values: "
                    f"{', '.join(lnk.shared_values[:6])}"
                )
    else:
        lines.append("\nCROSS-FILE LINKS: none found above confidence threshold.")
 
    # Fused numeric summaries
    if fused:
        lines.append(f"\nCROSS-FILE NUMERIC SUMMARIES ({len(fused)} records):")
        for rec in fused[:12]:
            lines.append(
                f"\n  [{rec['entity_type'].upper()}: {rec['entity_value']}]"
                f"  sources: {' + '.join(rec['source_files'])}"
                f"  (link confidence {rec['link_confidence']:.0%})"
            )
            for col, agg in list(rec["numeric_summary"].items())[:5]:
                lines.append(
                    f"    {col}: "
                    f"sum={agg['sum']}  mean={agg['mean']}  "
                    f"min={agg['min']}  max={agg['max']}"
                )
 
    # Prioritised insights
    if insights:
        lines.append(f"\nPRIORITISED CROSS-FILE INSIGHTS ({len(insights)} total):")
        for ins in insights[:10]:
            lines.append(f"\n  [{ins.priority.upper()}] {ins.title}")
            lines.append(f"    {ins.detail}")
 
    # LLM instruction block
    lines.append("\n-----------------------------------------------")
    lines.append("INSTRUCTIONS FOR REPORT GENERATION:")
    lines.append("-----------------------------------------------")
    if links:
        entity_types = list({lnk.entity_type for lnk in links})
        lines.append(
            f"  - The files share these entity dimensions: "
            f"{', '.join(entity_types)}"
        )
        lines.append(
            "  - You MUST compare and correlate data across files "
            "using these shared dimensions."
        )
        lines.append(
            "  - Use connecting language: "
            "'Taken alongside...', 'This is compounded by...', "
            "'However, this conflicts with...'"
        )
        lines.append(
            "  - Siloed per-file reporting is FORBIDDEN "
            "when shared entities exist."
        )
        lines.append(
            "  - Address all CRITICAL and HIGH priority insights above "
            "explicitly in the report."
        )
    else:
        lines.append(
            "  - No shared entity links found. "
            "Analyse each file independently but note any "
            "thematic connections where they exist."
        )
    lines.append("")
 
    return "\n".join(lines)
 
 
# ==============================================================================
# 9.  MAIN CLASS
# ==============================================================================
 
@dataclass
class FusionResult:
    cross_file_context: str
    ranked_files:       list[tuple[str, float]]
    links:              list[LinkRecord]
    insights:           list[InsightRecord]
    entity_map:         dict
    fused_records:      list[dict]
    summary_log:        str
 
 
class DataFusionAgent:
    """
    Cross-file intelligence layer for HPGPT.
 
    Usage:
        agent  = DataFusionAgent(file_data, user_prompt)
        result = agent.run()
 
    `file_data` is the dict already built by agent_orchestrator step 1:
        {clean_filename: data_dict_from_extract_data()}
 
    The agent reuses entities already extracted by data_agent -- no duplication.
    """
 
    def __init__(self, file_data: dict, user_prompt: str = ""):
        self.file_data    = file_data
        self.user_prompt  = user_prompt
 
    def run(self) -> FusionResult:
        n = len(self.file_data)
        logger.info(f"[DataFusion] Starting fusion for {n} file(s).")
 
        # -- Single file -- skip most steps ---------------------------------
        if n < 2:
            entity_map   = _build_entity_map(self.file_data)
            ranked_files = score_file_relevance(self.file_data, self.user_prompt)
            context      = self._single_file_context(entity_map)
            return FusionResult(
                cross_file_context = context,
                ranked_files       = ranked_files,
                links              = [],
                insights           = [],
                entity_map         = entity_map,
                fused_records      = [],
                summary_log        = f"Single file -- no cross-file analysis.",
            )
 
        # -- Step 1: File relevance ranking --------------------------------
        ranked_files = score_file_relevance(self.file_data, self.user_prompt)
 
        # -- Step 2: Entity map (reuses data_agent output) -----------------
        entity_map = _build_entity_map(self.file_data)
 
        # -- Step 3: Link discovery with confidence ------------------------
        links = _discover_links(entity_map)
        logger.info(
            f"[DataFusion] Links found: {len(links)} "
            f"(above confidence {_MIN_LINK_CONFIDENCE})"
        )
 
        # -- Step 4: Numeric fusion ----------------------------------------
        fused = _fuse_numerics(self.file_data, links) if links else []
        logger.info(f"[DataFusion] Fused numeric records: {len(fused)}")
 
        # -- Step 5: Insight prioritisation -------------------------------
        insights = _prioritise_insights(links, fused, self.file_data, entity_map)
        critical = sum(1 for i in insights if i.priority == "Critical")
        high     = sum(1 for i in insights if i.priority == "High")
        logger.info(
            f"[DataFusion] Insights: {len(insights)} total  "
            f"({critical} Critical, {high} High)"
        )
 
        # -- Step 6: Render context ----------------------------------------
        context = _render_context(
            self.file_data, entity_map,
            links, fused, insights, ranked_files
        )
 
        summary = (
            f"Files: {n} | Links: {len(links)} | "
            f"Fused records: {len(fused)} | "
            f"Insights: {len(insights)} "
            f"({critical} Critical, {high} High)"
        )
 
        return FusionResult(
            cross_file_context = context,
            ranked_files       = ranked_files,
            links              = links,
            insights           = insights,
            entity_map         = entity_map,
            fused_records      = fused,
            summary_log        = summary,
        )
 
    def _single_file_context(self, entity_map: dict) -> str:
        fname = next(iter(self.file_data))
        entities = entity_map.get(fname, {})
        if not entities:
            return ""
        lines = [f"ENTITY DIMENSIONS DETECTED in {fname}:"]
        for etype, info in entities.items():
            sv = ", ".join(info.get("sample_values", [])[:5])
            lines.append(f"  - {etype}: {sv}")
        return "\n".join(lines)