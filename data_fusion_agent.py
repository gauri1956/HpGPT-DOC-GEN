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
 
def _calculate_normalized_score(impact: int, confidence: int, entity_type: str) -> tuple[float, str]:
    # Normalize inputs to 1-5
    impact = max(1, min(5, int(impact)))
    confidence = max(1, min(5, int(confidence)))
    
    # Relevance mapping (1-5)
    relevance_map = {
        "product": 5,
        "department": 5,
        "region": 5,
        "location": 4,
        "employee": 4,
        "period": 2,
        "vendor": 3,
        "category": 3,
        "role": 3,
    }
    relevance = relevance_map.get(entity_type.lower(), 3)
    
    score = impact * confidence * relevance
    if score >= 75:
        priority = "Critical"  # Top Priority
    elif score >= 40:
        priority = "High"      # High Priority
    else:
        priority = "Medium"    # Medium Priority
        
    return float(score), priority


def _prioritise_insights(
    links: list[LinkRecord],
    fused: list[dict],
    file_data: dict,
    entity_map: dict,
) -> list[InsightRecord]:
    insights: list[InsightRecord] = []

    # -- Insight type A: link-level observations -------------------------------
    for link in links:
        # base impact of 2, plus 1 if link confidence > 70%
        impact = 3 if link.confidence >= 0.70 else 2
        confidence = int(round(link.confidence * 5))
        score, priority = _calculate_normalized_score(impact, confidence, link.entity_type)
        
        strength = "High" if priority in ("Critical", "High") else "Medium"
        
        if link.shared_values:
            sample_vals = ", ".join(link.shared_values[:4])
            detail = (
                f"{link.file_a} and {link.file_b} share "
                f"{len(link.shared_values)} common {link.entity_type} "
                f"values (e.g. {sample_vals}). "
                f"Link confidence: {link.confidence:.0%}. "
                f"Analyse these shared {link.entity_type}s for correlated trends. "
                f"(Insight Strength: {strength})"
            )
        else:
            detail = (
                f"{link.file_a} and {link.file_b} both contain a "
                f"{link.entity_type} dimension. "
                f"Align reporting on this shared axis. "
                f"(Insight Strength: {strength})"
            )

        title = (
            f"Cross-file {link.entity_type.title()} connection: "
            f"{link.file_a}  {link.file_b}"
        )
        insights.append(InsightRecord(
            title       = title,
            detail      = detail,
            priority    = priority,
            files       = [link.file_a, link.file_b],
            entity_type = link.entity_type,
            score       = score,
        ))

    # -- Insight type B: numeric fusion observations & contradiction detection -
    for rec in fused:
        if not rec.get("is_canonical_match", True):
            continue
        if not rec["numeric_summary"]:
            continue

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
                impact = 5 if rel_diff > 0.3 else 4
                confidence = 4
                score, priority = _calculate_normalized_score(impact, confidence, rec['entity_type'])
                strength = "Very High" if priority == "Critical" else "High"
                detail = (
                    f"Contradiction detected for {rec['entity_type']} '{rec['entity_value']}': "
                    f"'{key_a}' is {mean_a} in {rec['source_files'][0]}, but "
                    f"'{key_b}' is {mean_b} in {rec['source_files'][1]}. "
                    f"This is a variance of {diff:.1f} ({rel_diff:.1%}). "
                    f"Investigate the data sources for reporting discrepancies. "
                    f"(Insight Strength: {strength})"
                )
            else:
                title = f"Data Bridge: {rec['entity_type'].title()} '{rec['entity_value']}' numeric consistency"
                impact = 3
                confidence = 4
                score, priority = _calculate_normalized_score(impact, confidence, rec['entity_type'])
                detail = (
                    f"Consistent data found for {rec['entity_type']} '{rec['entity_value']}' across files: "
                    f"'{key_a}' is {mean_a} in {rec['source_files'][0]} and "
                    f"'{key_b}' is {mean_b} in {rec['source_files'][1]}. "
                    f"This confirms alignment on this shared metric. "
                    f"(Insight Strength: Medium)"
                )
                
            insights.append(InsightRecord(
                title       = title,
                detail      = detail,
                priority    = priority,
                files       = rec["source_files"],
                entity_type = rec["entity_type"],
                score       = score,
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
                pct = abs(t.get("pct_change", 0))
                impact = 5 if pct > 30 else 4 if pct > 15 else 3
                confidence = 4
                score, priority = _calculate_normalized_score(impact, confidence, "period")
                strength = "High" if priority in ("Critical", "High") else "Medium"
                
                detail = (
                    f"In {fname}, metric '{col}' is {t['trend_signal']} "
                    f"over period '{t['period_column']}'. "
                    f"(Insight Strength: {strength})"
                )
                insights.append(InsightRecord(
                    title       = title,
                    detail      = detail,
                    priority    = priority,
                    files       = [fname],
                    entity_type = "period",
                    score       = score,
                ))

    # -- Insight type D: Cross-file Correlation & Dependency Detection ---------
    # Group fused records by (file_a, file_b, ncol_a, ncol_b, entity_type)
    groups = defaultdict(list)
    for rec in fused:
        keys = list(rec["numeric_summary"].keys())
        if len(keys) < 2:
            continue
        key_a, key_b = keys[0], keys[1]
        
        ncol_a = re.sub(r'_in_[a-f0-9]+$', '', key_a)
        ncol_b = re.sub(r'_in_[a-f0-9]+$', '', key_b)
        
        file_a = rec["source_files"][0]
        file_b = rec["source_files"][1]
        
        group_key = (file_a, file_b, ncol_a, ncol_b, rec["entity_type"])
        groups[group_key].append(rec)
        
    for (file_a, file_b, ncol_a, ncol_b, etype), recs in groups.items():
        # Minimum sample check of 6 (prevent misleading correlations)
        if len(recs) < 6:
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
                
        if len(list_a) >= 6:
            s_a = pd.Series(list_a)
            s_b = pd.Series(list_b)
            if s_a.std() > 0 and s_b.std() > 0:
                r_coeff = s_a.corr(s_b)
                if not pd.isna(r_coeff) and abs(r_coeff) >= 0.75:
                    direction_str = "positive" if r_coeff > 0 else "negative"
                    
                    impact = int(round(abs(r_coeff) * 5))
                    confidence = 5 if len(list_a) >= 12 else 4 if len(list_a) >= 8 else 3
                    score, priority = _calculate_normalized_score(impact, confidence, etype)
                    strength = "Very High" if abs(r_coeff) >= 0.90 else "High"
                    
                    samples = []
                    for val, ma, mb in val_names[:3]:
                        samples.append(f"'{val}' ({ncol_a}={ma:.1f}, {ncol_b}={mb:.1f})")
                    sample_str = ", ".join(samples)
                    
                    title = f"Cross-file Correlation: '{ncol_a}' and '{ncol_b}' ({r_coeff:.2f})"
                    detail = (
                        f"A Pearson correlation of {r_coeff:.2f} was detected between "
                        f"'{ncol_a}' in {file_a} and '{ncol_b}' in {file_b} across the shared '{etype}' dimension. "
                        f"Sample pairings: {sample_str}. "
                        f"(Insight Strength: {strength})"
                    )
                    
                    insights.append(InsightRecord(
                        title       = title,
                        detail      = detail,
                        priority    = priority,
                        files       = [file_a, file_b],
                        entity_type = etype,
                        score       = score,
                    ))

    # -- Insight type E: In-Memory Inventory-Sales & Turnover Analyzer ----------
    # Load DataFrames in memory from file_data to avoid disk IO
    dfs = {}
    for fname, data in file_data.items():
        if "raw_df" in data:
            dfs[fname] = data["raw_df"]
        elif "sheets_df" in data:
            dfs.update(data["sheets_df"])
            
    sales_df = None
    inv_df = None
    sales_file = None
    inv_file = None

    for name, df in dfs.items():
        cols_lower = [str(c).lower() for c in df.columns]
        is_sales = any(c in ["petrol", "diesel", "lpg", "lubricants"] or "revenue" in c for c in cols_lower)
        is_inv = any("stock" in c or "inventory" in c or "issued" in c for c in cols_lower)
        
        if is_sales and not is_inv:
            sales_df = df
            sales_file = name
        elif is_inv:
            inv_df = df
            inv_file = name
            
    if sales_df is not None and inv_df is not None:
        sales_cols = {c.lower(): c for c in sales_df.columns}
        inv_cols = {c.lower(): c for c in inv_df.columns}
        
        prod_col = next((c for c in inv_df.columns if "product" in c.lower()), None)
        closing_col = next((inv_cols[c] for c in inv_cols if "closing" in c and "stock" in c), None)
        issued_col = next((inv_cols[c] for c in inv_cols if "issued" in c or "dispatch" in c or "quantity_issued" in c), None)
        
        if prod_col and closing_col and issued_col:
            grouped_inv = inv_df.groupby(prod_col).agg({
                closing_col: 'mean',
                issued_col: 'sum'
            })
            
            grouped_inv['turnover'] = grouped_inv[issued_col] / grouped_inv[closing_col]
            grouped_inv['daily_issued'] = grouped_inv[issued_col] / 30.0
            grouped_inv['days_stock'] = grouped_inv[closing_col] / grouped_inv['daily_issued']
            
            prod_stats = {}
            for prod, row in grouped_inv.iterrows():
                p_lower = str(prod).strip().lower()
                prod_stats[p_lower] = {
                    "turnover": row['turnover'],
                    "days_stock": row['days_stock'],
                    "mean_closing": row[closing_col],
                    "sum_issued": row[issued_col],
                    "raw_name": str(prod)
                }
                
            # Turnover Speed Insight
            if "petrol" in prod_stats and "lpg" in prod_stats:
                pet_to = prod_stats["petrol"]["turnover"]
                lpg_to = prod_stats["lpg"]["turnover"]
                if lpg_to > 0:
                    ratio = pet_to / lpg_to
                    score, priority = _calculate_normalized_score(4, 5, "product")
                    title = "Inventory Turnover Speed Discrepancy"
                    detail = (
                        f"Petrol inventory turnover ratio is {ratio:.1f}x faster than LPG "
                        f"(Petrol turnover: {pet_to:.2f}, LPG turnover: {lpg_to:.2f}). "
                        f"This indicates much faster stock clearance for Petrol compared to LPG. "
                        f"(Insight Strength: High)"
                    )
                    insights.append(InsightRecord(
                        title=title, detail=detail, priority=priority,
                        files=[inv_file], entity_type="product", score=score
                    ))
                    
            # Days of Stock Demand Coverage Insights
            for p_lower, stats in prod_stats.items():
                days = stats["days_stock"]
                if days > 0:
                    score, priority = _calculate_normalized_score(3, 5, "product")
                    title = f"Product Demand Coverage: {stats['raw_name']}"
                    detail = (
                        f"{stats['raw_name']} closing stock covers approximately {days:.1f} days of demand "
                        f"(based on average daily issued quantity of {stats['sum_issued']/30.0:.1f} units). "
                        f"(Insight Strength: Medium)"
                    )
                    insights.append(InsightRecord(
                        title=title, detail=detail, priority=priority,
                        files=[inv_file], entity_type="product", score=score
                    ))
                    
            # Stock Contribution vs Sales Volume Share Imbalance
            sales_p_cols = {}
            for col in sales_df.columns:
                col_l = col.lower()
                for p in ["petrol", "diesel", "lpg", "lubricants"]:
                    if p in col_l:
                        sales_p_cols[p] = col
                        
            stock_totals = grouped_inv[closing_col]
            total_stock_all = stock_totals.sum()
            
            sales_totals = {}
            for p, col in sales_p_cols.items():
                sales_totals[p] = sales_df[col].sum()
            total_sales_all = sum(sales_totals.values())
            
            if total_stock_all > 0 and total_sales_all > 0:
                for p in ["petrol", "diesel", "lpg", "lubricants"]:
                    inv_match_key = next((idx for idx in stock_totals.index if p in str(idx).lower()), None)
                    if inv_match_key and p in sales_totals:
                        stock_share = stock_totals[inv_match_key] / total_stock_all
                        sales_share = sales_totals[p] / total_sales_all
                        
                        if stock_share > sales_share + 0.10:
                            # Potential overstock! Compare growth percentages
                            sales_col_name = sales_p_cols[p]
                            sales_vals = sales_df[sales_col_name].dropna().tolist()
                            sales_growth = 0.0
                            if len(sales_vals) >= 2 and sales_vals[0] > 0:
                                sales_growth = (sales_vals[-1] - sales_vals[0]) / sales_vals[0] * 100.0
                                
                            time_col = next((inv_cols[c] for c in inv_cols if "month" in c or "period" in c or "date" in c), None)
                            stock_growth = 0.0
                            if time_col:
                                p_rows = inv_df[inv_df[prod_col] == inv_match_key].sort_values(by=time_col)
                                if len(p_rows) >= 2:
                                    stock_vals = p_rows[closing_col].dropna().tolist()
                                    if stock_vals[0] > 0:
                                        stock_growth = (stock_vals[-1] - stock_vals[0]) / stock_vals[0] * 100.0
                                        
                            score, priority = _calculate_normalized_score(5, 5, "product")
                            title = f"Potential Overstock Risk: {inv_match_key}"
                            detail = (
                                f"{inv_match_key} contributes {stock_share:.1%} of total closing stock, but only "
                                f"{sales_share:.1%} of sales volume. Potential overstock risk detected because "
                                f"{inv_match_key} closing stock grew by {stock_growth:.1f}% while sales grew by only "
                                f"{sales_growth:.1f}%. (Insight Strength: Very High)"
                            )
                            insights.append(InsightRecord(
                                title=title, detail=detail, priority=priority,
                                files=[inv_file, sales_file], entity_type="product", score=score
                            ))

    # Sort: Critical -> High -> Medium -> Low, then by score desc
    _ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    insights.sort(key=lambda i: (_ORDER.get(i.priority, 3), -i.score))
    return insights
 
 
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
        for rec in fused[:2]:
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
        for ins in insights[:3]:
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