import os
import re
import uuid
import time
import logging
from collections import Counter
 
from data_agent     import extract_data, detect_content_intent
from analysis_agent import run_analysis, run_insight
from graph_agent    import generate_charts_from_data, generate_charts_from_markers, dedup_charts
from doc_writer     import generate_docx
from pdf_writer     import generate_pdf
from ppt_writer     import generate_ppt
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
LLM_CALL_DELAY = 8
 
# ── Document-type detection from user prompt ──────────────────────────────────
# Maps keyword signals in the user prompt to a structured document type.
# This runs BEFORE any file analysis so the output format is known early.
 
_DOCTYPE_SIGNALS = {
    "training_material": [
        'training material', 'training document', 'training module',
        'learning material', 'educational content', 'course material',
        'onboarding document', 'induction material',
    ],
    "sop": [
        'sop', 'standard operating procedure', 'operating procedure',
        'process document', 'step-by-step procedure', 'work instruction',
    ],
    "policy_document": [
        'policy document', 'policy', 'guidelines', 'compliance document',
        'hr policy', 'code of conduct', 'rulebook',
    ],
    "business_report": [
        'business report', 'performance report', 'analytical report',
        'kpi report', 'executive report', 'management report',
        'quarterly report', 'annual report',
    ],
    "financial_report": [
        'financial report', 'finance report', 'p&l', 'profit and loss',
        'balance sheet', 'cash flow', 'financial analysis', 'revenue report',
    ],
}
 
 
def _detect_doc_type_from_prompt(prompt: str) -> str | None:
    """
    Detect explicit document type from the user's prompt.
    Returns a doc_type string or None (let intent detection handle it).
    """
    pl = prompt.lower()
    for doc_type, signals in _DOCTYPE_SIGNALS.items():
        if any(sig in pl for sig in signals):
            logger.info(f"Doc type detected from prompt: {doc_type}")
            return doc_type
    return None
 
 
def _clean_name(raw_name):
    """Strip UUID prefix from saved filename."""
    return re.sub(r'^[a-f0-9]{8,}_', '', raw_name)
 
 
def _display_name(name):
    """Human-readable version of filename for headings."""
    return (name.replace('.xlsx', '').replace('.csv', '').replace('.pdf', '')
                .replace('.docx', '').replace('.txt', '').replace('_', ' ').strip())
 
 
def run_agent_from_api(prompt, file_paths=None, file_path=None,
                       output_format="docx", has_files=None,
                       add_acknowledgement=False):
    """
    Main pipeline entry point.
 
    Args:
        prompt              : user's text prompt
        file_paths          : list of uploaded file paths
        file_path           : single file path (legacy support)
        output_format       : "docx" | "pdf" | "pptx"
        has_files           : bool — explicitly passed from app.py.
                              If None, inferred from file_paths.
        add_acknowledgement : bool — add acknowledgement slide (pptx only)
 
    Returns:
        For pptx  → (output_path, slide_count)
        For others → output_path
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
 
    # Normalise inputs
    if file_paths is None: file_paths = []
    if file_path and file_path not in file_paths:
        file_paths.append(file_path)
    file_paths = [fp for fp in file_paths if fp and os.path.exists(fp)]
 
    if has_files is None:
        has_files = len(file_paths) > 0
 
    logger.info(f"Files received: {len(file_paths)} → {[os.path.basename(f) for f in file_paths]}")
    logger.info(f"has_files={has_files} | output_format={output_format} | "
                f"add_acknowledgement={add_acknowledgement}")
 
    # ── Detect document type from user prompt ─────────────────────────────────
    # This is the primary hook for issue #8 (document-type detection).
    # If the user says "create a training material" or "write an SOP", we know
    # the output format before touching any file.
    doc_type = _detect_doc_type_from_prompt(prompt)
    logger.info(f"Prompt-detected doc_type: {doc_type}")
 
    seen_signatures = set()
    ordered_charts  = []
 
    # ── 1. Extract data from all files ────────────────────────────────────────
    file_data = {}
    for fp in file_paths:
        raw_name   = os.path.basename(fp)
        clean_name = _clean_name(raw_name)
        logger.info(f"Extracting: {clean_name}")
        data         = extract_data(fp)
        data['file'] = clean_name
        file_data[clean_name] = data
 
        dtype  = data.get('type', 'unknown')
        rows   = data.get('row_count', 'N/A')
        charts = len(data.get('chart_candidates', []))
        logger.info(f"  type={dtype} | rows={rows} | chart_candidates={charts}")
 
        if dtype == 'multi_sheet':
            for sname, sdata in data.get('sheets', {}).items():
                logger.info(f"    sheet={sname} | rows={sdata.get('row_count','?')} | "
                            f"candidates={len(sdata.get('chart_candidates',[]))}")
 
    # ── 1b. Detect intent across all files ───────────────────────────────────
    file_intents = {}
    for name, data in file_data.items():
        intent = detect_content_intent(data, user_prompt=prompt)
        file_intents[name] = intent
        logger.info(f"Intent for {name}: {intent}")
 
    dominant_intent = Counter(file_intents.values()).most_common(1)[0][0] if file_intents else "informational"
    logger.info(f"Dominant intent: {dominant_intent}")
 
    # ── 1c. Resolve effective doc_type ───────────────────────────────────────
    # If the prompt didn't name a type, try to infer from intent.
    # This mapping covers the most common analytical/operational cases.
    if doc_type is None:
        _intent_to_dtype = {
            "analytical":   "business_report",
            "operational":  "operational",
            "educational":  "educational",
            "policy":       "policy_document",
            "informational": None,  # keep None — don't over-classify
        }
        doc_type = _intent_to_dtype.get(dominant_intent)
        if doc_type:
            logger.info(f"Doc type inferred from intent: {doc_type}")
 
    # ── 2. Pre-compute deterministic data-charts per file ─────────────────────
    data_charts_by_file = {}
    total_data_charts = 0
    for name, data in file_data.items():
        if file_intents.get(name, "analytical") in ("analytical", "operational"):
            charts = generate_charts_from_data(data)
        else:
            charts = []
        data_charts_by_file[name] = charts
        total_data_charts += len(charts)
        logger.info(f"Data charts from {name}: {len(charts)}")
    logger.info(f"Total data charts: {total_data_charts}")
 
    # ── 3. Generate title ─────────────────────────────────────────────────────
    title = _make_title(prompt, list(file_data.keys()), dominant_intent)
    time.sleep(LLM_CALL_DELAY)
 
    # ── 4. Build document body ────────────────────────────────────────────────
    if not has_files:
        from llm_agent import query_llama
        logger.info("No files — pure LLM generation (noinput mode)")
        raw_body = query_llama(
            prompt,
            output_type=output_format,
            has_files=False
        )
 
        body, body_charts = generate_charts_from_markers(raw_body)
        deduped, seen_signatures = dedup_charts(body_charts, seen_signatures)
        ordered_charts.extend(deduped)
        logger.info(f"Charts kept for no-files generation: {len(deduped)} "
                    f"(of {len(body_charts)})")
 
    else:
        # ── 4a. Per-file fact digests ─────────────────────────────────────────
        digest_results = []
 
        for name, data in file_data.items():
            dtype  = data.get('type', 'unknown')
            intent = file_intents.get(name, "analytical")
            logger.info(f"Digesting [{dtype}] (intent={intent}): {name}")
 
            if dtype == 'multi_sheet':
                flat = _flatten_multisheet(data)
                digest_text, detected_intent = run_analysis(
                    flat,
                    instruction=f"{prompt} — analysing {_display_name(name)}",
                    intent=intent
                )
            else:
                digest_text, detected_intent = run_analysis(
                    data,
                    instruction=f"{prompt} — analysing {_display_name(name)}",
                    intent=intent
                )
 
            clean_digest, marker_charts = generate_charts_from_markers(digest_text)
 
            if intent in ("analytical", "operational"):
                file_charts = data_charts_by_file.get(name, []) + marker_charts
            else:
                file_charts = marker_charts
 
            deduped, seen_signatures = dedup_charts(file_charts, seen_signatures)
            ordered_charts.extend(deduped)
            logger.info(f"Charts kept for {name}: {len(deduped)} "
                        f"(data={len(data_charts_by_file.get(name, []))}, "
                        f"markers={len(marker_charts)})")
 
            digest_results.append((clean_digest, detected_intent))
            time.sleep(LLM_CALL_DELAY)
 
        # ── 4b. Single integrated report from all digests ─────────────────────
        logger.info(f"Generating integrated {output_format} report from "
                    f"{len(digest_results)} digest(s) "
                    f"(intent={dominant_intent}, doc_type={doc_type})...")
 
        # Pass doc_type into run_insight so output structure adapts to the
        # document type, not just the data intent.
        raw_report, _intent = run_insight(
            digest_results,
            prompt,
            output_format=output_format,
            doc_type=doc_type         # ← KEY: drives output structure
        )
        clean_report, report_charts = generate_charts_from_markers(raw_report)
 
        deduped, seen_signatures = dedup_charts(report_charts, seen_signatures)
        ordered_charts.extend(deduped)
        logger.info(f"Charts kept for integrated report: {len(deduped)} "
                    f"(of {len(report_charts)})")
 
        body = clean_report
 
    # ── 5. Verify chart paths ─────────────────────────────────────────────────
    logger.info(f"Total charts to embed: {len(ordered_charts)}")
    verified = [(h, p) for h, p, _sig in ordered_charts if os.path.exists(p)]
    logger.info(f"Verified chart paths: {len(verified)}")
 
    # ── 6. Write document ─────────────────────────────────────────────────────
    output_path = _resolve_path(_sanitize(title, output_format))
 
    if output_format == "docx":
        generate_docx(body, output_path, title, chart_images=verified)
        return output_path
 
    elif output_format == "pdf":
        generate_pdf(body, output_path, title, chart_images=verified)
        return output_path
 
    elif output_format == "pptx":
        slide_count = generate_ppt(
            body,
            output_path,
            filename_title=title,
            chart_images=verified,
            add_acknowledgement=add_acknowledgement
        )
        logger.info(f"PPT saved: {output_path} ({slide_count} slides)")
        return output_path, slide_count
 
    else:
        raise ValueError(f"Unsupported format: {output_format}")
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def _flatten_multisheet(data):
    """Merge multi-sheet data into a single summary dict for the LLM."""
    sheets         = data.get('sheets', {})
    all_candidates = []
    all_samples    = {}
    all_stats      = {}
    all_columns    = []
 
    for sheet_name, sdata in sheets.items():
        all_candidates.extend(sdata.get('chart_candidates', []))
        all_columns.extend(sdata.get('columns', []))
        if sdata.get('sample'):
            all_samples[sheet_name] = sdata['sample']
        if sdata.get('stats'):
            all_stats[sheet_name]   = sdata['stats']
 
    return {
        "file":             data.get('file', ''),
        "type":             "tabular",
        "columns":          list(dict.fromkeys(all_columns)),
        "sheets_summary":   {s: {"rows": d.get('row_count'), "columns": d.get('columns')}
                             for s, d in sheets.items()},
        "sample":           all_samples,
        "stats":            all_stats,
        "chart_candidates": all_candidates[:16]
    }
 
 
# ── Title generation ──────────────────────────────────────────────────────────
 
_TITLE_FILLER = {
    'give', 'me', 'a', 'an', 'the', 'make', 'create', 'generate',
    'write', 'build', 'produce', 'get', 'please', 'can', 'you',
    'i', 'want', 'need', 'help', 'with', 'for', 'my', 'our',
}
 
_FORCE_UPPER = {'hpcl', 'bpcl', 'iocl', 'lpg', 'atf', 'kl', 'mt',
                'omc', 'hr', 'fy', 'kpi', 'pdf', 'ppt', 'ai', 'api'}
 
 
def _title_case(text: str) -> str:
    minor = {'and', 'or', 'the', 'of', 'in', 'on', 'with', 'a',
             'an', 'to', 'for', 'at', 'by', 'from', 'into', 'as'}
    words = text.strip().split()
    result = []
    for i, w in enumerate(words):
        low = w.lower()
        if low in _FORCE_UPPER:
            result.append(low.upper())
        elif i == 0 or i == len(words) - 1 or low not in minor:
            result.append(w.capitalize())
        else:
            result.append(low)
    return ' '.join(result)
 
 
def _fallback_title(prompt: str) -> str:
    words = re.sub(r'[^\w\s]', '', prompt).strip().split()
    words = [w for w in words if w.lower() not in _TITLE_FILLER]
    raw   = ' '.join(words[:7]) or 'HPCL Document'
    return _title_case(raw)
 
 
def _make_title(prompt: str, file_names: list, intent: str = "informational") -> str:
    from llm_agent import query_llama
 
    intent_hint = {
        "analytical":    "This is an analytical report — title should reflect analysis/insights.",
        "informational": "This is an informational document — title should reflect the topic covered.",
        "educational":   "This is an educational/training document — title should reflect the learning topic.",
        "policy":        "This is a policy/procedure document — title should reflect the policy subject.",
        "operational":   "This is an operational report — title should reflect the operational area.",
    }.get(intent, "")
 
    try:
        raw = query_llama(
            f"Give a short 5-7 word professional document title for this request: "
            f"{prompt[:200]}.\n"
            f"{intent_hint}\n"
            f"The title must reflect what the USER asked for, not just the file name.\n"
            f"Reply with ONLY the title — no quotes, no punctuation at the end, no explanation.",
            output_type="plan",
            has_files=bool(file_names)
        )
        title = raw.strip().strip('"\'').strip('.')
        return _title_case(title) if title else _fallback_title(prompt)
 
    except Exception as e:
        logger.warning(f"Title LLM call failed ({e}), using fallback.")
        return _fallback_title(prompt)
 
 
def _sanitize(text, ext):
    base = re.sub(r'[^\w\s-]', '', text).strip().lower()
    base = re.sub(r'[\s-]+', '_', base) or uuid.uuid4().hex[:8]
    return f"outputs/{base}.{ext}"
 
 
def _resolve_path(path):
    """Avoid overwriting existing files — append (2), (3) etc."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}({i}){ext}"):
        i += 1
    return f"{base}({i}){ext}"
 