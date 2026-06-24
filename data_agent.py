"""
data_agent.py  -  HPGPT Data Extraction & Intent Detection
===========================================================
Improvements vs previous version
----------------------------------
1.  Entity extraction passes actual DataFrame so real values are captured,
    not just column names.
2.  Multi-sheet Excel: entities aggregated across ALL sheets; cross-sheet
    overlaps surfaced in cross_sheet_entities key.
3.  detect_content_intent() returns (intent, confidence) tuple.
    Confidence is 0-1; callers that only need the string unpack the first element.
4.  _ENTITY_DIMENSIONS now uses the same synonym table as data_fusion_agent
    so canonical names are consistent everywhere.
"""
 
import os
import re
import json
import logging
import pandas as pd
 
logger = logging.getLogger(__name__)
 
ABBREVS = {'HPCL', 'BPCL', 'IOCL', 'LPG', 'ATF', 'KL', 'MT', 'OMC', 'HP', 'INR'}
 
 
def _clean_col(name):
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = (name.replace('KL', ' KL').replace('MT', ' MT')
                .replace('INR', ' INR').replace('Lakh', ' Lakh'))
    name = name.replace('_', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    parts = name.split()
    return ' '.join(p if p.upper() in ABBREVS else p.title() for p in parts)
 
 
def _is_id_column(series, col_name):
    """Detect useless ID/code columns that shouldn't be charted."""
    col_lower = col_name.lower()
    id_keywords = ['id', 'code', 'number', 'no', 'num', 'index',
                   'serial', 'sr', 'rank']
    if any(k in col_lower.split() or col_lower.startswith(k)
           for k in id_keywords):
        return True
    try:
        vals = series.dropna().astype(float).tolist()
        if len(vals) < 2:
            return False
        diffs = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
        if all(d == 1.0 for d in diffs):
            return True
        if (all(d == diffs[0] for d in diffs)
                and max(vals) < 10_000 and min(vals) > 0):
            return True
    except Exception:
        pass
    return False
 
 
# -- Synonym table (mirrors data_fusion_agent for consistency) -----------------
_ENTITY_DIMENSIONS: dict[str, list[str]] = {
    "department":   ["department", "dept", "division", "business unit",
                     "function", "vertical", "bu"],
    "region":       ["region", "zone", "area", "geography", "territory",
                     "cluster", "circle"],
    "location":     ["location", "city", "state", "branch", "office",
                     "plant", "site", "depot", "terminal", "hub"],
    "product":      ["product", "item", "sku", "material", "commodity",
                     "fuel", "grade"],
    "employee":     ["employee", "emp", "staff", "worker", "personnel",
                     "headcount", "associate"],
    "vendor":       ["vendor", "supplier", "contractor", "agency"],
    "customer":     ["customer", "client", "account", "buyer",
                     "dealer", "distributor"],
    "category":     ["category", "type", "class", "segment", "group",
                     "tier"],
    "project":      ["project", "initiative", "program", "scheme"],
    "period":       ["month", "quarter", "year", "period", "fy",
                     "date", "week", "fiscal"],
    "cost_center":  ["cost center", "cost centre", "profit center",
                     "cost code"],
    "asset":        ["asset", "equipment", "machine", "facility",
                     "tank", "vehicle"],
    "team":         ["team", "unit", "squad"],
    "role":         ["role", "designation", "title", "position",
                     "grade", "level", "band"],
}
 
 
def _col_entity_type(col_name: str) -> str | None:
    cl = col_name.lower().strip()
    for entity, keywords in _ENTITY_DIMENSIONS.items():
        if any(kw in cl for kw in keywords):
            return entity
    return None
 
 
# -- Enhancement 1: Entity extraction with actual values -----------------------
 
def extract_entities(data: dict, df: pd.DataFrame = None) -> dict:
    """
    Extract business entity dimensions from column names.
    When `df` is provided, sample_values are extracted from actual data.
 
    Returns:
        {entity_type: {"columns": [...], "sample_values": [...]}}
    """
    if data.get("type") == "multi_sheet":
        col_lists = [
            s.get("columns", [])
            for s in data.get("sheets", {}).values()
        ]
    else:
        col_lists = [data.get("columns", [])]
 
    entities: dict[str, dict] = {}
 
    for cols in col_lists:
        for col in cols:
            etype = _col_entity_type(col)
            if etype is None:
                continue
 
            if etype not in entities:
                entities[etype] = {"columns": [], "sample_values": []}
 
            if col not in entities[etype]["columns"]:
                entities[etype]["columns"].append(col)
 
            # Enhancement 1 fix: pull real values from the DataFrame
            if df is not None and col in df.columns:
                vals = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .unique()
                    .tolist()
                )
                existing = set(entities[etype]["sample_values"])
                entities[etype]["sample_values"] = list(
                    existing | set(vals)
                )[:20]
 
    return entities
 
 
# -- Enhancement 4: Confidence-based intent detection --------------------------
 
def detect_content_intent(
    data: dict, user_prompt: str = ""
) -> tuple[str, float]:
    """
    Returns (intent, confidence) where confidence is 0.0-1.0.
 
    intent   {"analytical", "informational", "educational",
               "policy", "operational"}
 
    Backward-compatible: callers that previously assigned the return value
    to a single variable will get the tuple; callers that unpack work fine.
    """
    scores: dict[str, float] = {
        "analytical":    0.0,
        "informational": 0.0,
        "educational":   0.0,
        "policy":        0.0,
        "operational":   0.0,
    }
 
    prompt_lower = user_prompt.lower()
 
    # 1. Prompt signals -- strongest weight
    prompt_signals = {
        "educational":   ['educat', 'teach', 'learn', 'course',
                          'training material', 'lesson', 'curriculum',
                          'syllabus', 'tutorial'],
        "policy":        ['policy', 'guideline', 'procedure', 'rule',
                          'compliance', 'sop', 'standard operating'],
        "operational":   ['inventory', 'stock', 'log', 'transaction',
                          'dispatch', 'receipt', 'operation'],
        "informational": ['information', 'overview', 'about',
                          'introduction', 'describe', 'explain what'],
        "analytical":    ['analysis', 'report', 'kpi', 'performance',
                          'insight', 'dashboard', 'trend', 'forecast'],
    }
    for intent, kws in prompt_signals.items():
        if any(w in prompt_lower for w in kws):
            scores[intent] += 0.6
 
    # 2. File type signals
    file_type = data.get('type', 'unknown')
    file_name = data.get('file', '').lower()
 
    if file_type == 'text':
        content = data.get('content', '').lower()
        if any(w in content for w in
               ['objective', 'module', 'chapter', 'topic', 'learn']):
            scores["educational"] += 0.2
        if any(w in content for w in
               ['shall', 'must', 'procedure', 'policy', 'guideline']):
            scores["policy"] += 0.2
        scores["informational"] += 0.1
 
    # 3. Column-name signals
    columns: list[str] = []
    if file_type == 'multi_sheet':
        for sdata in data.get('sheets', {}).values():
            columns.extend(sdata.get('columns', []))
    else:
        columns = data.get('columns', [])
 
    col_str = ' '.join(columns).lower()
 
    col_signals: dict[str, tuple[list[str], int]] = {
        "operational": (
            ['stock', 'inventory', 'dispatch', 'receipt', 'quantity',
             'issued', 'received', 'balance', 'transaction', 'date',
             'opening', 'closing', 'batch', 'lot', 'serial'], 2),
        "educational": (
            ['topic', 'module', 'chapter', 'score', 'marks', 'grade',
             'student', 'course', 'subject', 'lesson', 'question'], 2),
        "policy": (
            ['policy', 'rule', 'guideline', 'compliance', 'regulation',
             'procedure', 'standard', 'requirement', 'category',
             'type', 'description'], 2),
        "analytical": (
            ['revenue', 'sales', 'profit', 'loss', 'cost', 'expense',
             'attrition', 'headcount', 'kpi', 'performance', 'target',
             'actual', 'variance', 'growth', 'rate', 'ratio', 'margin',
             'salary', 'budget', 'forecast', 'quarter', 'annual',
             'monthly'], 2),
    }
 
    numeric_cols: list[str] = data.get('numeric_cols', [])
    if file_type == 'multi_sheet':
        for sdata in data.get('sheets', {}).values():
            numeric_cols.extend(sdata.get('numeric_cols', []))
 
    for intent, (kws, threshold) in col_signals.items():
        hits = sum(1 for k in kws if k in col_str)
        if hits >= threshold:
            scores[intent] += min(0.3, hits * 0.05)
 
    if len(numeric_cols) >= 3:
        scores["analytical"] += 0.15
 
    # 4. File name fallback
    fname_signals = {
        "analytical":  ['kpi', 'sales', 'revenue', 'hr', 'attrition',
                        'performance', 'report', 'dashboard'],
        "operational": ['stock', 'inventory', 'dispatch', 'log'],
        "policy":      ['policy', 'procedure', 'guideline', 'sop'],
        "educational": ['course', 'training', 'education', 'module'],
    }
    for intent, kws in fname_signals.items():
        if any(w in file_name for w in kws):
            scores[intent] += 0.1
 
    # Pick winner
    best_intent = max(scores, key=scores.__getitem__)
    best_score  = scores[best_intent]
 
    if best_score == 0.0:
        best_intent = "informational"
        best_score  = 0.3
 
    confidence = round(min(best_score, 1.0), 2)
 
    if confidence < 0.5:
        logger.warning(
            f"[data_agent] Low-confidence intent '{best_intent}' "
            f"({confidence}) for {data.get('file', '')}. "
            f"All scores: {scores}"
        )
    else:
        logger.info(
            f"[data_agent] intent='{best_intent}' confidence={confidence} "
            f"for {data.get('file', '')}"
        )
 
    return best_intent, confidence
 
 
# -- File readers ---------------------------------------------------------------
 
def _read_csv(path):
    return pd.read_csv(path)
 
 
def _read_excel(path):
    xl = pd.ExcelFile(path)
    if len(xl.sheet_names) == 1:
        return xl.parse(xl.sheet_names[0])
    return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
 
 
def _read_text(path):
    with open(path, 'r', errors='ignore') as f:
        return f.read()
 
 
def _read_json_tabular(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = json.load(f)
 
    records = None
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        records = raw
    elif isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                records = v
                break
 
    if records is None:
        return None
    return pd.json_normalize(records)
 
 
# -- Main entry point -----------------------------------------------------------
 
def extract_data(file_path, instruction=""):
    ext  = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)
    name = re.sub(r'^[a-f0-9]{8,}_', '', name)
 
    try:
        if ext == ".csv":
            df = _read_csv(file_path)
            return {"file": name, "type": "tabular", **_summarize_df(df, name)}
 
        elif ext in (".xlsx", ".xls"):
            result = _read_excel(file_path)
            if isinstance(result, dict):
                # Enhancement 2: full multi-sheet with cross-sheet entity merge
                return {
                    "file": name,
                    "type": "multi_sheet",
                    **_summarize_multisheet(result, name)
                }
            return {"file": name, "type": "tabular",
                    **_summarize_df(result, name)}
 
        elif ext == ".json":
            try:
                df = _read_json_tabular(file_path)
            except Exception as e:
                logger.warning(f"JSON parse failed for {name}: {e}")
                df = None
            if df is not None and not df.empty:
                return {"file": name, "type": "tabular",
                        **_summarize_df(df, name)}
            from input_handler import parse_file_only
            text = parse_file_only(file_path, ext)
            return {"file": name, "type": "text", "content": text[:8000]}
 
        elif ext in (".txt", ".md"):
            return {"file": name, "type": "text",
                    "content": _read_text(file_path)[:8000]}
 
        else:
            from input_handler import parse_file_only
            text = parse_file_only(file_path, ext)
            return {"file": name, "type": "text", "content": text[:8000]}
 
    except Exception as e:
        logger.error(f"data_agent failed on {file_path}: {e}")
        return {"file": name, "type": "error", "error": str(e)}
 
 
# -- Single-frame summarizer ----------------------------------------------------

_DOMAIN_KEYWORDS = {
    "HR": ["employee", "attrition", "salary", "headcount", "recruitment", "training", "wages", "staff", "resignation", "turnover", "role", "designation", "grade"],
    "Finance": ["revenue", "sales", "profit", "loss", "cost", "expense", "budget", "p&l", "margin", "earnings", "price", "billing", "inr"],
    "Operations": ["inventory", "stock", "dispatch", "receipt", "quantity", "production", "plant", "depot", "terminal", "warehouse", "capacity", "volume", "kl", "mt"],
}


def parse_period_to_sort_key(val):
    if pd.isnull(val):
        return (0, 0, 0)
    s = str(val).strip().lower()
    
    # 1. Look for a 4-digit year (e.g. 2024) or 2-digit year preceded by FY/CY (e.g. FY24, FY 24)
    year = 0
    year_match = re.search(r'\b(20\d{2})\b', s)
    if year_match:
        year = int(year_match.group(1))
    else:
        # Check for FY24 or FY 24 or '24
        fy_match = re.search(r'\b(?:fy|cy)?\s*(\d{2})\b', s)
        if fy_match:
            year = 2000 + int(fy_match.group(1))
            
    # 2. Look for quarters: q1, q2, q3, q4
    quarter = 0
    q_match = re.search(r'\bq([1-4])\b', s)
    if q_match:
        quarter = int(q_match.group(1))
    else:
        # Check for "quarter 1" or "qtr 1"
        q_match2 = re.search(r'\b(?:quarter|qtr)\s*([1-4])\b', s)
        if q_match2:
            quarter = int(q_match2.group(1))

    # 3. Look for months: jan, feb...
    month = 0
    month_map = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
        "aug": 8, "august": 8, "sep": 9, "september": 9, "oct": 10, "october": 10,
        "nov": 11, "november": 11, "dec": 12, "december": 12
    }
    for mname, mval in month_map.items():
        if mname in s:
            month = mval
            break
            
    return (year, quarter, month)


def detect_trend_direction(df: pd.DataFrame, period_col: str, numeric_col: str) -> dict | None:
    if df.empty or len(df) < 2:
        return None
    try:
        sorted_df = df.copy()
        
        # Drop rows with null values in period_col or numeric_col
        sorted_df = sorted_df.dropna(subset=[period_col, numeric_col])
        if len(sorted_df) < 2:
            return None
            
        period_vals = sorted_df[period_col].astype(str).str.strip().str.lower()
        
        # 1. Try standard pd.to_datetime first
        converted = pd.to_datetime(sorted_df[period_col], errors='coerce')
        if converted.notnull().mean() > 0.5:
            sorted_df["__period_sort__"] = converted
            sorted_df = sorted_df.dropna(subset=["__period_sort__"]).sort_values("__period_sort__")
        else:
            # 2. Try parsing custom period tuples (year, quarter, month)
            sorted_df["__period_sort__"] = sorted_df[period_col].apply(parse_period_to_sort_key)
            has_sort_info = sorted_df["__period_sort__"].apply(lambda x: x != (0, 0, 0)).any()
            if has_sort_info:
                sorted_df = sorted_df.sort_values("__period_sort__")
            else:
                # 3. Fallback to numeric sorting if possible
                numeric_periods = pd.to_numeric(sorted_df[period_col], errors='coerce')
                if numeric_periods.notnull().mean() > 0.5:
                    sorted_df["__period_sort__"] = numeric_periods
                    sorted_df = sorted_df.dropna(subset=["__period_sort__"]).sort_values("__period_sort__")
                else:
                    # Fallback: keep original order
                    pass

        y = sorted_df[numeric_col].astype(float).tolist()
        if len(y) >= 2:
            n = len(y)
            x = list(range(n))
            mean_x = sum(x) / n
            mean_y = sum(y) / n
            
            num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
            den = sum((x[i] - mean_x) ** 2 for i in range(n))
            slope = num / den if den != 0 else 0.0
            
            pct_change = ((y[-1] - y[0]) / y[0] * 100) if y[0] != 0 else 0.0
            
            # Stricter trend validation:
            if abs(pct_change) < 0.5:
                direction = "flat"
            elif slope > 0 and y[-1] >= y[0]:
                direction = "up"
            elif slope < 0 and y[-1] <= y[0]:
                direction = "down"
            elif pct_change > 0:
                direction = "up"
            else:
                direction = "down"
                
            return {
                "direction": direction,
                "slope": round(slope, 3),
                "pct_change": round(pct_change, 1),
                "period_column": period_col,
                "start_value": round(y[0], 2),
                "end_value": round(y[-1], 2),
                "trend_signal": f"trending {direction} (from {round(y[0], 1)} to {round(y[-1], 1)}, change: {round(pct_change, 1)}%)"
            }
    except Exception as e:
        logger.warning(f"Trend detection failed for {period_col} and {numeric_col}: {e}")
    return None


def _generate_metadata_heuristics(df: pd.DataFrame, useful_numeric: list[str]) -> tuple[dict, list[str], list[str]]:
    # 1. Domain Map
    domain_map = {}
    for col in df.columns:
        cl = col.lower()
        matched = False
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            if any(kw in cl for kw in keywords):
                domain_map[col] = domain
                matched = True
                break
        if not matched and col in useful_numeric:
            domain_map[col] = "Finance"  # Default domain for numeric cols if unmatched
            
    # 2. Causal Hints
    causal_hints = []
    has_attrition = any("attrition" in c or "resignation" in c or "turnover" in c for c in df.columns.str.lower())
    has_salary = any("salary" in c or "compensation" in c or "pay" in c or "wages" in c for c in df.columns.str.lower())
    has_engagement = any("engagement" in c or "satisfaction" in c or "score" in c for c in df.columns.str.lower())
    has_sales = any("sales" in c or "revenue" in c or "volume" in c for c in df.columns.str.lower())
    has_profit = any("profit" in c or "margin" in c or "net" in c for c in df.columns.str.lower())
    has_inventory = any("inventory" in c or "stock" in c for c in df.columns.str.lower())
    has_dispatch = any("dispatch" in c or "receipt" in c or "issued" in c or "received" in c for c in df.columns.str.lower())
    has_training = any("training" in c or "learn" in c or "course" in c for c in df.columns.str.lower())
    has_performance = any("performance" in c or "rating" in c or "score" in c for c in df.columns.str.lower())

    if has_attrition and has_salary:
        causal_hints.append("Lower salary levels or compensation grades are often a primary driver of high attrition/resignation rates.")
    if has_attrition and has_engagement:
        causal_hints.append("Employee engagement/satisfaction scores are a leading indicator of resignation and attrition trends.")
    if has_sales and has_profit:
        causal_hints.append("Sales volumes and transaction counts are the direct drivers of overall revenue and profit margins.")
    if has_inventory and has_dispatch:
        causal_hints.append("Operational dispatch/issue rates directly drawdown opening stock levels, determining closing stock.")
    if has_training and has_performance:
        causal_hints.append("Structured training hours and curriculum completion rates influence employee performance ratings.")

    # 3. Contradiction Seeds
    contradiction_seeds = []
    for col in df.columns:
        cl = col.lower()
        if "attrition" in cl or "resignation" in cl or "turnover" in cl:
            contradiction_seeds.append(f"Verify if the attrition/turnover rate metric '{col}' matches corresponding columns or exit counts in other uploaded datasets.")
        elif "sales" in cl or "revenue" in cl or "profit" in cl:
            contradiction_seeds.append(f"Verify if Sales/Revenue/Profit totals for '{col}' match across different segment or financial reports.")
        elif "stock" in cl or "inventory" in cl:
            contradiction_seeds.append(f"Verify if opening/closing stock levels in '{col}' align with dispatch, receipts, or logistics records in other files.")
            
    return domain_map, causal_hints, contradiction_seeds


def _summarize_df(df: pd.DataFrame, file_name: str = "") -> dict:
    df = df.dropna(how='all').reset_index(drop=True)

    col_map = {c: _clean_col(str(c)) for c in df.columns}
    df = df.rename(columns=col_map)

    numeric_cols   = df.select_dtypes(include='number').columns.tolist()
    text_cols      = df.select_dtypes(exclude='number').columns.tolist()
    useful_numeric = [c for c in numeric_cols
                      if not _is_id_column(df[c], c)]

    df[numeric_cols] = df[numeric_cols].round(2)

    stats: dict = {}
    for col in useful_numeric:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min":  round(float(df[col].min()),  2),
            "max":  round(float(df[col].max()),  2),
            "std":  round(float(df[col].std()),  2),
        }

    chart_candidates: list[dict] = []
    for tcol in text_cols[:2]:
        for ncol in useful_numeric[:5]:
            labels = df[tcol].astype(str).tolist()[:12]
            values = df[ncol].dropna().round(2).tolist()[:12]
            if len(labels) == len(values) and len(values) >= 2:
                chart_candidates.append({
                    "label_col": tcol,
                    "value_col": ncol,
                    "labels":    labels,
                    "values":    values,
                })

    # Enhancement 1: pass actual df so values are captured
    partial = {"type": "tabular", "columns": df.columns.tolist()}
    entities = extract_entities(partial, df=df)

    # 1. Compute entity-grouped statistics on full data
    entity_stats = []
    for etype, info in entities.items():
        for ecol in info["columns"]:
            if ecol in df.columns:
                for ncol in useful_numeric:
                    try:
                        grouped = df.groupby(ecol)[ncol].agg(['mean', 'min', 'max', 'sum', 'count'])
                        stats_dict = {}
                        for val, row in grouped.iterrows():
                            val_str = str(val).strip()
                            if val_str and val_str not in ("nan", "None", ""):
                                stats_dict[val_str] = {
                                    "mean": round(float(row['mean']), 2) if pd.notnull(row['mean']) else 0.0,
                                    "min": round(float(row['min']), 2) if pd.notnull(row['min']) else 0.0,
                                    "max": round(float(row['max']), 2) if pd.notnull(row['max']) else 0.0,
                                    "sum": round(float(row['sum']), 2) if pd.notnull(row['sum']) else 0.0,
                                    "count": int(row['count'])
                                }
                        if stats_dict:
                            entity_stats.append({
                                "entity_type": etype,
                                "entity_col": ecol,
                                "numeric_col": ncol,
                                "stats": stats_dict
                            })
                    except Exception as e:
                        logger.warning(f"Failed to compute entity stats for {ecol} and {ncol}: {e}")

    # 2. Identify period column and compute trend direction
    period_col = None
    for col in df.columns:
        if _col_entity_type(col) == "period":
            period_col = col
            break
            
    trends = {}
    if period_col:
        for col in useful_numeric:
            t = detect_trend_direction(df, period_col, col)
            if t:
                trends[col] = t
                
    # 3. Generate domain_map, causal_hints, and contradiction_seeds
    domain_map, causal_hints, contradiction_seeds = _generate_metadata_heuristics(df, useful_numeric)

    return {
        "columns":          df.columns.tolist(),
        "row_count":        len(df),
        "numeric_cols":     useful_numeric,
        "text_cols":        text_cols,
        "sample":           df.head(5).to_dict(orient='records'),
        "stats":            stats,
        "entities":         entities,
        "dataset_type":     file_name,
        "chart_candidates": chart_candidates,
        "entity_stats":     entity_stats,
        "trends":           trends,
        "domain_map":       domain_map,
        "causal_hints":     causal_hints,
        "contradiction_seeds": contradiction_seeds,
    }
 
 
# -- Enhancement 2: Multi-sheet summarizer with cross-sheet entity merge --------
 
def _summarize_multisheet(sheets_dict: dict, file_name: str = "") -> dict:
    """
    Summarize each sheet individually, then merge entity maps across
    all sheets so cross-sheet relationships are captured.
    """
    sheets_summary: dict[str, dict]  = {}
    all_entities:   dict[str, dict]  = {}
 
    for sheet_name, df in sheets_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
 
        summary = _summarize_df(df, sheet_name)
        sheets_summary[sheet_name] = summary
 
        # Merge entity maps
        for etype, info in summary.get("entities", {}).items():
            if etype not in all_entities:
                all_entities[etype] = {
                    "columns":       [],
                    "sample_values": [],
                    "sheets":        [],
                }
            for col in info["columns"]:
                if col not in all_entities[etype]["columns"]:
                    all_entities[etype]["columns"].append(col)
            existing = set(all_entities[etype]["sample_values"])
            all_entities[etype]["sample_values"] = list(
                existing | set(info["sample_values"])
            )[:20]
            if sheet_name not in all_entities[etype]["sheets"]:
                all_entities[etype]["sheets"].append(sheet_name)
 
    # Enhancement 2: flag entities shared across multiple sheets
    cross_sheet_entities = {
        etype: info
        for etype, info in all_entities.items()
        if len(info.get("sheets", [])) > 1
    }
    if cross_sheet_entities:
        logger.info(
            f"[data_agent] {file_name} -- cross-sheet entities: "
            f"{list(cross_sheet_entities.keys())}"
        )
 
    return {
        "sheets":               sheets_summary,
        "entities":             all_entities,
        "cross_sheet_entities": cross_sheet_entities,
        "file":                 file_name,
    }
 