import os
import re
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

ABBREVS = {'HPCL','BPCL','IOCL','LPG','ATF','KL','MT','OMC','HP','INR'}

def _clean_col(name):
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', name)
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    name = name.replace('KL',' KL').replace('MT',' MT').replace('INR',' INR').replace('Lakh',' Lakh')
    name = name.replace('_', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    parts = name.split()
    return ' '.join(p if p.upper() in ABBREVS else p.title() for p in parts)


def _is_id_column(series, col_name):
    """Detect useless ID/code columns that shouldn't be charted."""
    col_lower = col_name.lower()
    id_keywords = ['id', 'code', 'number', 'no', 'num', 'index', 'serial', 'sr', 'rank']
    if any(k in col_lower.split() or col_lower.startswith(k) for k in id_keywords):
        return True
    try:
        vals = series.dropna().astype(float).tolist()
        if len(vals) < 2:
            return False
        diffs = [abs(vals[i+1] - vals[i]) for i in range(len(vals)-1)]
        if all(d == 1.0 for d in diffs):
            return True
        if all(d == diffs[0] for d in diffs) and max(vals) < 10000 and min(vals) > 0:
            return True
    except Exception:
        pass
    return False


# ── Enhancement 1: Entity Extraction ─────────────────────────────────────────
# Maps canonical entity names to keyword fragments matched against column names.
# More specific keywords listed first to avoid false matches.

_ENTITY_DIMENSIONS = {
    "department":   ["department", "dept", "division", "function", "business unit"],
    "region":       ["region", "zone", "area", "geography", "territory"],
    "location":     ["location", "city", "state", "branch", "office", "plant", "site"],
    "product":      ["product", "item", "sku", "material", "commodity", "fuel", "grade"],
    "employee":     ["employee", "emp", "staff", "worker", "personnel"],
    "vendor":       ["vendor", "supplier", "contractor", "agency"],
    "customer":     ["customer", "client", "account", "buyer"],
    "category":     ["category", "type", "class", "segment", "group"],
    "project":      ["project", "initiative", "program", "scheme"],
    "period":       ["month", "quarter", "year", "period", "fy", "date", "week"],
    "cost_center":  ["cost center", "cost centre", "profit center"],
    "asset":        ["asset", "equipment", "machine", "facility", "tank"],
    "team":         ["team", "unit", "squad", "cluster"],
}


def _col_entity_type(col_name: str) -> str | None:
    """Return canonical entity type for a column name, or None."""
    cl = col_name.lower().strip()
    for entity, keywords in _ENTITY_DIMENSIONS.items():
        if any(kw in cl for kw in keywords):
            return entity
    return None


def extract_entities(data: dict, df: pd.DataFrame = None) -> dict:
    """
    Enhancement 1 — Entity Extraction with actual values.

    Returns:
        {entity_type: {"columns": [...], "sample_values": [...]}}

    If `df` is provided, sample_values are extracted from actual data rows.
    Otherwise falls back to column-name-only detection (no values).
    """
    # Collect columns from all sheets or single frame
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

            # Enhancement 1: pull actual distinct values from the DataFrame
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


# ── Enhancement 4: Confidence-Based Intent Detection ─────────────────────────

def detect_content_intent(data: dict, user_prompt: str = "") -> tuple[str, float]:
    """
    Enhancement 4 — returns (intent, confidence_score).

    intent is one of:
        "analytical" | "informational" | "educational" | "policy" | "operational"

    confidence_score is 0.0–1.0:
        >= 0.8  → strong signal, use directly
        0.5–0.8 → moderate, use with logging
        < 0.5   → weak, log a warning and consider fallback

    Callers that previously expected only a string still work because Python
    allows: intent, _ = detect_content_intent(data, prompt)
    For backward compat, agent_orchestrator.py unpacks the tuple.
    """
    scores: dict[str, float] = {
        "analytical":    0.0,
        "informational": 0.0,
        "educational":   0.0,
        "policy":        0.0,
        "operational":   0.0,
    }

    prompt_lower = user_prompt.lower()

    # ── 1. User prompt — strong signal (weight 0.6) ───────────────────────────
    prompt_signals = {
        "educational":   ['educat', 'teach', 'learn', 'course', 'training material',
                          'lesson', 'curriculum', 'syllabus', 'tutorial'],
        "policy":        ['policy', 'guideline', 'procedure', 'rule', 'compliance',
                          'sop', 'standard operating'],
        "operational":   ['inventory', 'stock', 'log', 'transaction', 'dispatch',
                          'receipt', 'operation'],
        "informational": ['information', 'overview', 'about', 'introduction',
                          'describe', 'explain what'],
        "analytical":    ['analysis', 'report', 'kpi', 'performance', 'insight',
                          'dashboard', 'trend', 'forecast'],
    }
    for intent, kws in prompt_signals.items():
        if any(w in prompt_lower for w in kws):
            scores[intent] += 0.6

    # ── 2. File type signals (weight 0.2) ─────────────────────────────────────
    file_type = data.get('type', 'unknown')
    file_name = data.get('file', '').lower()

    if file_type == 'text':
        content = data.get('content', '').lower()
        if any(w in content for w in ['objective', 'module', 'chapter', 'topic', 'learn']):
            scores["educational"] += 0.2
        if any(w in content for w in ['shall', 'must', 'procedure', 'policy', 'guideline']):
            scores["policy"] += 0.2
        scores["informational"] += 0.1   # text files lean informational by default

    # ── 3. Column-name signals (weight 0.15 per keyword match, capped at 0.3) ─
    columns = []
    if file_type == 'multi_sheet':
        for sdata in data.get('sheets', {}).values():
            columns.extend(sdata.get('columns', []))
    else:
        columns = data.get('columns', [])

    col_str = ' '.join(columns).lower()

    col_signals = {
        "operational":   (['stock', 'inventory', 'dispatch', 'receipt', 'quantity',
                           'issued', 'received', 'balance', 'transaction', 'date',
                           'opening', 'closing', 'batch', 'lot', 'serial'], 2),
        "educational":   (['topic', 'module', 'chapter', 'score', 'marks', 'grade',
                           'student', 'course', 'subject', 'lesson', 'question'], 2),
        "policy":        (['policy', 'rule', 'guideline', 'compliance', 'regulation',
                           'procedure', 'standard', 'requirement', 'category',
                           'type', 'description'], 2),
        "analytical":    (['revenue', 'sales', 'profit', 'loss', 'cost', 'expense',
                           'attrition', 'headcount', 'kpi', 'performance', 'target',
                           'actual', 'variance', 'growth', 'rate', 'ratio', 'margin',
                           'salary', 'budget', 'forecast', 'quarter', 'annual',
                           'monthly'], 2),
    }
    numeric_cols = data.get('numeric_cols', [])
    if file_type == 'multi_sheet':
        for sdata in data.get('sheets', {}).values():
            numeric_cols.extend(sdata.get('numeric_cols', []))

    for intent, (kws, threshold) in col_signals.items():
        hits = sum(1 for k in kws if k in col_str)
        if hits >= threshold:
            scores[intent] += min(0.3, hits * 0.05)

    # Numeric-heavy files lean analytical
    if len(numeric_cols) >= 3:
        scores["analytical"] += 0.15

    # ── 4. File name fallback (weight 0.1) ────────────────────────────────────
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

    # ── 5. Pick winner ────────────────────────────────────────────────────────
    best_intent = max(scores, key=scores.__getitem__)
    best_score  = scores[best_intent]

    # If nothing scored, default to informational with low confidence
    if best_score == 0.0:
        best_intent = "informational"
        best_score  = 0.3

    # Normalise score to 0–1 (max possible raw score ≈ 1.05)
    confidence = round(min(best_score / 1.0, 1.0), 2)

    if confidence < 0.5:
        logger.warning(
            f"[data_agent] Low-confidence intent '{best_intent}' "
            f"({confidence}) for {data.get('file', '?')}. "
            f"Scores: {scores}"
        )
    else:
        logger.info(
            f"[data_agent] Intent='{best_intent}' confidence={confidence} "
            f"for {data.get('file', '?')}"
        )

    return best_intent, confidence


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
                # Multi-sheet — Enhancement 2
                return {"file": name, "type": "multi_sheet",
                        **_summarize_multisheet(result, name)}
            else:
                return {"file": name, "type": "tabular",
                        **_summarize_df(result, name)}

        elif ext == ".json":
            try:
                df = _read_json_tabular(file_path)
            except Exception as e:
                logger.warning(f"JSON parse failed for {name}: {e}")
                df = None
            if df is not None and not df.empty:
                return {"file": name, "type": "tabular", **_summarize_df(df, name)}
            from input_handler import parse_file_only
            text = parse_file_only(file_path, ext)
            return {"file": name, "type": "text", "content": text[:8000]}

        elif ext in (".txt", ".md"):
            return {"file": name, "type": "text", "content": _read_text(file_path)[:8000]}

        else:
            from input_handler import parse_file_only
            text = parse_file_only(file_path, ext)
            return {"file": name, "type": "text", "content": text[:8000]}

    except Exception as e:
        logger.error(f"data_agent failed on {file_path}: {e}")
        return {"file": name, "type": "error", "error": str(e)}


def _summarize_df(df: pd.DataFrame, file_name: str = "") -> dict:
    df = df.dropna(how='all').reset_index(drop=True)

    col_map = {c: _clean_col(str(c)) for c in df.columns}
    df = df.rename(columns=col_map)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    text_cols    = df.select_dtypes(exclude='number').columns.tolist()
    useful_numeric = [c for c in numeric_cols if not _is_id_column(df[c], c)]

    df[numeric_cols] = df[numeric_cols].round(2)

    stats = {}
    for col in useful_numeric:
        stats[col] = {
            "mean": round(float(df[col].mean()), 2),
            "min":  round(float(df[col].min()),  2),
            "max":  round(float(df[col].max()),  2),
            "std":  round(float(df[col].std()),  2)
        }

    chart_candidates = []
    for tcol in text_cols[:2]:
        for ncol in useful_numeric[:5]:
            labels = df[tcol].astype(str).tolist()[:12]
            values = df[ncol].dropna().round(2).tolist()[:12]
            if len(labels) == len(values) and len(values) >= 2:
                chart_candidates.append({
                    "label_col": tcol,
                    "value_col": ncol,
                    "labels":    labels,
                    "values":    values
                })

    # Enhancement 1 fix: pass actual df so entity values are extracted
    partial_data = {"type": "tabular", "columns": df.columns.tolist()}
    entities = extract_entities(partial_data, df=df)

    return {
        "columns":          df.columns.tolist(),
        "row_count":        len(df),
        "numeric_cols":     useful_numeric,
        "text_cols":        text_cols,
        "sample":           df.head(5).to_dict(orient='records'),
        "stats":            stats,
        "entities":         entities,
        "dataset_type":     file_name,
        "chart_candidates": chart_candidates
    }


# ── Enhancement 2: Multi-Sheet Excel with cross-sheet entity aggregation ──────

def _summarize_multisheet(sheets_dict: dict, file_name: str = "") -> dict:
    """
    Enhancement 2 — summarizes each sheet AND aggregates common entities
    across all sheets so cross-sheet relationships are visible.
    """
    sheets_summary = {}
    all_entities: dict[str, dict] = {}

    for sheet_name, df in sheets_dict.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        summary = _summarize_df(df, sheet_name)
        sheets_summary[sheet_name] = summary

        # Merge entity maps across sheets
        for etype, info in summary.get("entities", {}).items():
            if etype not in all_entities:
                all_entities[etype] = {"columns": [], "sample_values": [], "sheets": []}

            for col in info["columns"]:
                if col not in all_entities[etype]["columns"]:
                    all_entities[etype]["columns"].append(col)

            existing_vals = set(all_entities[etype]["sample_values"])
            all_entities[etype]["sample_values"] = list(
                existing_vals | set(info["sample_values"])
            )[:20]

            if sheet_name not in all_entities[etype]["sheets"]:
                all_entities[etype]["sheets"].append(sheet_name)

    # Log cross-sheet entity overlaps
    cross_sheet_entities = {
        etype: info
        for etype, info in all_entities.items()
        if len(info["sheets"]) > 1
    }
    if cross_sheet_entities:
        logger.info(
            f"[data_agent] {file_name} — cross-sheet entities: "
            f"{list(cross_sheet_entities.keys())}"
        )

    return {
        "sheets":               sheets_summary,
        "entities":             all_entities,
        "cross_sheet_entities": cross_sheet_entities,  # Enhancement 2
        "file":                 file_name,
    }