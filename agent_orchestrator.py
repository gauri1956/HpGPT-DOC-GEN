import os
import re
import uuid
import time
import logging
from collections import Counter
 
from data_agent        import extract_data, detect_content_intent
from analysis_agent    import run_analysis, run_insight
from graph_agent       import generate_charts_from_data, generate_charts_from_markers, dedup_charts
from doc_writer        import generate_docx
from pdf_writer        import generate_pdf
from ppt_writer        import generate_ppt
from data_fusion_agent import DataFusionAgent
from planner_agent     import plan_tasks   # ← NEW: import updated planner
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
LLM_CALL_DELAY = 8
 
# -- Document-type detection from user prompt ----------------------------------
_DOCTYPE_SIGNALS = {
    # --- Official government/PSU document types ---
    "file_note": [
        'file note', 'filenote', 'noting', 'office note',
        'note for file', 'note on file', 'note sheet', 'noting sheet',
    ],
    "office_memorandum": [
        'office memorandum', 'office memo', 'o.m.', 'om',
        'official memorandum', 'internal memorandum', 'memorandum', 'memo',
    ],
    "office_notice": [
        'office notice', 'public notice', 'circular notice',
        'official notice', 'staff notice', 'issue a notice',
        'draft a notice', 'write a notice', 'generate a notice',
    ],
    "circular": [
        'circular', 'office circular', 'departmental circular',
        'policy circular', 'administrative circular',
    ],
    "purchase_note": [
        'purchase note', 'procurement note', 'purchase order note',
        'indent note', 'purchase proposal', 'procurement proposal',
        'purchase request', 'procurement request',
        'purchase of ', 'procurement of ',
    ],
    "office_order": [
        'office order', 'administrative order', 'transfer order',
        'posting order', 'promotion order',
    ],
    # --- Standard analytical/content document types ---
    "training_material": [
        'training material', 'training document', 'training module',
        'learning material', 'educational content', 'course material',
        'onboarding document', 'induction material',
    ],
    "sop": [
        'sop', 'standard operating procedure', 'operating procedure',
        'process document', 'step-by-step procedure', 'work instruction',
        'office procedure manual', 'procedure manual', 'office manual',
        'work instruction manual',
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
 
_OFFICIAL_DOC_TYPES = {
    "file_note", "office_memorandum", "office_notice",
    "circular", "purchase_note", "office_order"
}
 
_HIGH_PRIORITY_DOC_TYPES = {
    "file_note", "office_memorandum", "office_notice",
    "circular", "purchase_note", "office_order",
    "business_report", "financial_report", "sop"
}

_MIN_SECTION_LENGTHS = {
    "business_report": {"## Executive Summary": 150, "## Key Findings": 300},
    "purchase_note":   {"## 1. Purpose": 200, "## 3. Operational": 200},
}
 
 
def _match_signal(pl: str, sig: str) -> bool:
    sig_stripped = sig.strip()
    if not sig_stripped:
        return False
    words = sig_stripped.split()
    if not words:
        return False
    escaped_words = []
    for i, w in enumerate(words):
        if i == len(words) - 1 and w[-1].isalnum():
            w_lower = w.lower()
            if w_lower.endswith('y'):
                base = re.escape(w[:-1])
                escaped_words.append(f"(?:{base}y|{base}ies)")
            elif w_lower.endswith('um'):
                base = re.escape(w[:-2])
                escaped_words.append(f"(?:{base}um|{base}ums|{base}a)")
            elif any(w_lower.endswith(sfx) for sfx in ('s', 'x', 'z', 'ch', 'sh')):
                escaped_words.append(f"{re.escape(w)}(?:es)?")
            else:
                escaped_words.append(f"{re.escape(w)}s?")
        else:
            escaped_words.append(re.escape(w))
    escaped_sig = r'\s+'.join(escaped_words)
    pattern = ""
    if sig_stripped[0].isalnum():
        pattern += r'\b'
    pattern += escaped_sig
    if sig_stripped[-1].isalnum():
        pattern += r'\b'
    return bool(re.search(pattern, pl))
 
 
def _detect_doc_type_from_prompt(prompt: str) -> str | None:
    pl = prompt.lower()
    matches = []
    for doc_type, signals in _DOCTYPE_SIGNALS.items():
        for sig in signals:
            if _match_signal(pl, sig):
                is_high = doc_type in _HIGH_PRIORITY_DOC_TYPES
                matches.append((doc_type, len(sig.strip()), is_high))
    if matches:
        matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
        detected = matches[0][0]
        logger.info(f"Doc type detected from prompt: {detected} (matches: {matches})")
        return detected
    return None
 
 
def _resolve_official_format(doc_type: str, subject: str) -> str | None:
    if doc_type not in _OFFICIAL_DOC_TYPES:
        return None
    from llm_agent import get_doctype_rules
    doc_type_labels = {
        "file_note":          "File Note",
        "office_memorandum":  "Office Memorandum (O.M.)",
        "office_notice":      "Office Notice",
        "circular":           "Office Circular",
        "purchase_note":      "Purchase / Procurement Note",
        "office_order":       "Office Order",
    }
    label = doc_type_labels.get(doc_type, doc_type.replace("_", " ").title())
    format_spec = get_doctype_rules(doc_type)
    result = (
        f"\n\nOFFICIAL DOCUMENT FORMAT — {label.upper()} — MANDATORY:\n"
        f"The document being generated is a '{label}'. "
        f"You MUST follow the official standard format below EXACTLY. "
        f"Do NOT use a generic report or essay structure. "
        f"Do NOT add sections that are not part of this format.\n\n"
        f"Standard format specification:\n"
        f"{format_spec}\n\n"
        f"ENFORCEMENT RULES:\n"
        f"- Every section listed in the format above MUST appear in the output with substantive content.\n"
        f"- Sections must appear in the exact order specified.\n"
        f"- For fields where the value is not known from context, use clean underlines "
        f"(e.g. 'Reference No.: __________________' or 'Date: __________________').\n"
        f"- Use the official field labels (e.g. 'Subject:', 'Reference:', 'To:', 'Through:') "
        f"exactly as they appear in the format.\n"
        f"- Do NOT convert header fields into ## section headings.\n"
        f"- Body sections (## headings) MUST contain substantive numbered paragraphs -- not one-liners.\n"
    )
    logger.info(f"[FormatResolver] Format resolved from document rules ({len(result)} chars)")
    return result
 
 
def _clean_name(raw_name):
    return re.sub(r'^[a-f0-9]{8,}_', '', raw_name)
 
 
def _display_name(name):
    return (name.replace('.xlsx', '').replace('.csv', '').replace('.pdf', '')
                .replace('.docx', '').replace('.txt', '').replace('_', ' ').strip())
 
 
def _get_section_contents(body: str) -> dict[str, str]:
    """
    Parse a markdown body and extract content for each heading.
    Maps a normalized heading (e.g., '## executive summary') to its text content.
    """
    sections = {}
    lines = body.splitlines()
    cur_heading = None
    cur_content = []
    
    for line in lines:
        s = line.strip()
        if s.startswith("#"):
            if cur_heading:
                sections[cur_heading] = "\n".join(cur_content).strip()
            cur_heading = s.lower()
            cur_content = []
        else:
            if cur_heading:
                cur_content.append(line)
                
    if cur_heading:
        sections[cur_heading] = "\n".join(cur_content).strip()
        
    return sections


def _validate_body(body: str, file_data: dict, cross_file_insights: list = None,
                   title: str = None, doc_type: str = None, prompt: str = "",
                   has_files: bool = True) -> list[str]:
    """
    Validate generated body text and return list of warning strings.
    Returns empty list if all checks pass.
    """
    warnings = []
 
    if not body or len(body.strip()) < 200:
        warnings.append("Validation Warning: Generated body is too short (< 200 chars). Provide more detailed instructions.")

    if doc_type and doc_type in _MIN_SECTION_LENGTHS:
        sections_dict = _get_section_contents(body)
        thresholds = _MIN_SECTION_LENGTHS[doc_type]
        for threshold_heading, min_len in thresholds.items():
            norm_threshold = threshold_heading.lower().strip()
            found_section_text = None
            found_heading_name = None
            
            for h_name, content_text in sections_dict.items():
                if norm_threshold in h_name:
                    found_section_text = content_text
                    found_heading_name = h_name
                    break
            
            if found_section_text is not None:
                actual_len = len(found_section_text)
                if actual_len < min_len:
                    warnings.append(
                        f"Validation Warning: Section '{threshold_heading}' content is too short ({actual_len} chars < {min_len} minimum). Provide more detailed analysis."
                    )
            else:
                warnings.append(
                    f"Validation Warning: Required section '{threshold_heading}' is missing or has no content."
                )
 
    if doc_type not in _OFFICIAL_DOC_TYPES and '##' not in body:
        warnings.append("Validation Warning: No section headings (##) found in output. Structure the document with appropriate headers.")
 
    generic_titles = ["hpcl document", "untitled document", "report", "learning module",
                      "employee training program", "development program", "kpi dashboard",
                      "training presentation"]
    if title and any(gt == title.lower().strip() for gt in generic_titles):
        warnings.append(f"Validation Warning: Document title '{title}' appears generic.")
 
    if doc_type not in _OFFICIAL_DOC_TYPES:
        placeholders = ['[insert', '[tbd]', '[todo]', '[placeholder]', 'lorem ipsum']
        for ph in placeholders:
            if ph in body.lower():
                warnings.append(f"Validation Warning: Placeholder text '{ph}' found.")
 
    if doc_type not in _OFFICIAL_DOC_TYPES and file_data:
        for fname, data in file_data.items():
            dname = _display_name(fname).lower()
            cols = []
            if data.get('type') == 'multi_sheet':
                for s in data.get('sheets', {}).values():
                    cols.extend(s.get('columns', []))
            else:
                cols = data.get('columns', [])
            top_cols = [c.lower() for c in cols[:5]]
            found = (dname in body.lower() or
                     any(c in body.lower() for c in top_cols if len(c) > 4))
            if not found:
                warnings.append(f"Validation Warning: File '{fname}' is not referenced in the output.")
 
    if len(file_data) > 1 and doc_type not in _OFFICIAL_DOC_TYPES:
        connecting_words = ["compare", "correlation", "contradict", "aligned", "alongside",
                            "compounded", "across", "versus", "vs", "difference"]
        if not any(w in body.lower() for w in connecting_words):
            warnings.append("Validation Warning: No cross-file comparison terms found.")
        if cross_file_insights:
            referenced_insights = 0
            for ins in cross_file_insights:
                words = re.findall(r'\b\w{4,}\b', ins.title.lower() + " " + ins.detail.lower())
                stop_words = {"file", "dataset", "report", "connection", "numeric", "variance",
                              "insight", "priority", "critical", "high", "medium", "low",
                              "value", "values", "common", "shared", "trends", "trend"}
                keywords = [w for w in words if w not in stop_words]
                if keywords:
                    hits = sum(1 for kw in keywords if kw in body.lower())
                    if hits >= min(1, len(keywords)):
                        referenced_insights += 1
            if referenced_insights == 0:
                warnings.append("Validation Warning: None of the prioritized cross-file insights were detected in the generated body.")
 
    recs_match = re.search(r'##\s*Recommendations\b.*', body, re.IGNORECASE | re.DOTALL)
    if recs_match:
        recs_text = recs_match.group()
        if not any(char.isdigit() for char in recs_text):
            warnings.append("Validation Warning: Recommendations section lacks numeric data or specific metrics.")
 
    if doc_type == "business_report":
        recs_match = re.search(r'##\s*(?:Strategic\s+)?Recommendations\b.*', body, re.IGNORECASE | re.DOTALL)
        if recs_match:
            recs_text = recs_match.group()
            missing_structure = [
                term for term in ["Recommendation", "Operational Rationale", "Expected Outcome & Metrics"]
                if term.lower() not in recs_text.lower()
            ]
            if missing_structure:
                warnings.append(
                    f"Validation Warning: Strategic recommendations missing structured sections: {missing_structure}."
                )
 
    if doc_type in _OFFICIAL_DOC_TYPES:
        if "subject:" not in body.lower() and not any("subject:" in s.lower() for s in body.splitlines()[:15]):
            warnings.append("Validation Warning: Official document missing required 'Subject:' header.")
 
        sig_indicators = ["prepared by", "sd/-", "approved by", "recommended by", "submitted for approval"]
        if not any(ind in body.lower() for ind in sig_indicators):
            warnings.append("Validation Warning: No approval authority or signatory block detected.")
 
        if doc_type in ["file_note", "office_memorandum", "circular"]:
            bullet_count = body.count("\n- ") + body.count("\n* ")
            if bullet_count > 5:
                warnings.append(
                    f"Validation Warning: Official doc '{doc_type}' should use numbered paragraphs, not bullet points."
                )
 
        if doc_type == "purchase_note":
            template_leftovers = [
                "write 3-5 numbered paragraphs", "explain what exactly",
                "provide a markdown table", "for unknown values",
                "write 2-3 numbered paragraphs", "write a clear, formal recommendation"
            ]
            found_leftovers = [tl for tl in template_leftovers if tl in body.lower()]
            if found_leftovers:
                warnings.append(
                    f"Validation Warning: Purchase Note contains template instruction text: {found_leftovers}."
                )
 
        mandatory_checks = {
            "file_note": [
                ("## background / facts", "File Note missing 'Background / Facts of the Case' section."),
                ("## analysis / deliberations", "File Note missing 'Analysis / Deliberations' section."),
                ("## recommendation", "File Note missing 'Recommendation' section."),
                ("## approval sought", "File Note missing 'Approval Sought' section."),
                ("submitted for approval", "File Note missing 'Submitted for Approval' signature block.")
            ],
            "office_memorandum": [
                ("no.:", "Office Memorandum missing 'No.:' in header."),
                ("from:", "Office Memorandum missing 'From:' in header."),
                ("to:", "Office Memorandum missing 'To:' in header."),
                ("## body", "Office Memorandum missing mandatory 'Body' section."),
                ("sd/-", "Office Memorandum missing 'Sd/-' signature marker.")
            ],
            "office_notice": [
                ("notice no.:", "Office Notice missing 'Notice No.:' in header."),
                ("## notice body", "Office Notice missing mandatory 'Notice Body' section."),
                ("## compliance", "Office Notice missing mandatory 'Compliance / Action Required' section."),
                ("by order", "Office Notice missing 'By Order' authority line."),
                ("distribution:", "Office Notice missing 'Distribution:' list.")
            ],
            "circular": [
                ("circular no.", "Circular missing 'Circular No.' in header."),
                ("## preamble", "Circular missing mandatory 'Preamble / Background' section."),
                ("## instructions", "Circular missing mandatory 'Instructions / Directives' section."),
                ("## effective date", "Circular missing mandatory 'Effective Date' section."),
                ("distribution:", "Circular missing 'Distribution:' list.")
            ],
            "purchase_note": [
                ("## 1. purpose", "Purchase Note missing 'Purpose and Operational Requirement' section."),
                ("## 2. specification", "Purchase Note missing 'Specification of Items / Services' section."),
                ("## 3. operational justification", "Purchase Note missing 'Operational Justification' section."),
                ("## 4. budget provision", "Purchase Note missing 'Budget Provision' section."),
                ("## 5. procurement method", "Purchase Note missing 'Procurement Method' section."),
                ("## 6. proposed delivery", "Purchase Note missing 'Proposed Delivery Schedule' section."),
                ("## 7. recommendation", "Purchase Note missing 'Recommendation and Approval Sought' section.")
            ],
            "office_order": [
                ("office order no.", "Office Order missing 'Office Order No.' in header."),
                ("## order", "Office Order missing mandatory 'Order' section."),
                ("## effective date", "Office Order missing mandatory 'Effective Date' section."),
                ("## compliance", "Office Order missing mandatory 'Compliance' section."),
                ("copy to:", "Office Order missing 'Copy to:' distribution list.")
            ]
        }
        for term, msg in mandatory_checks.get(doc_type, []):
            if term not in body.lower():
                warnings.append(f"Validation Warning: {msg}")
 
        honorific_match = re.search(
            r'\b(shri|smt|mr|ms|dr)\.?\s+[a-zA-Z]{2,}\s+[a-zA-Z]{2,}',
            body, re.IGNORECASE
        )
        if honorific_match:
            warnings.append(
                f"Validation Warning: Potential invented person name: '{honorific_match.group()}'."
            )
 
        date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b', body)
        for dt in date_patterns:
            if dt not in prompt:
                in_files = any(dt in str(fdata) for fdata in file_data.values())
                if not in_files:
                    warnings.append(f"Validation Warning: Potential invented date: '{dt}'.")
 
    if doc_type not in _OFFICIAL_DOC_TYPES:
        for fname, data in file_data.items():
            trends = data.get("trends", {})
            if data.get("type") == "multi_sheet":
                trends = {}
                for sname, sdata in data.get("sheets", {}).items():
                    if sdata.get("trends"):
                        trends.update(sdata["trends"])
            for col, t in trends.items():
                direction = t.get("direction")
                if direction not in ("up", "down"):
                    continue
                clean_col = col.lower().replace("_", " ")
                body_lower = body.lower()
                if len(clean_col) > 3 and clean_col in body_lower:
                    idx = 0
                    while True:
                        idx = body_lower.find(clean_col, idx)
                        if idx == -1:
                            break
                        start_win = max(0, idx - 150)
                        end_win = min(len(body_lower), idx + len(clean_col) + 150)
                        window = body_lower[start_win:end_win]
                        if direction == "up":
                            negatives = ["decrease", "decline", "drop", "fell", "downward", "shrank", "reduced"]
                            if any(neg in window for neg in negatives) and not any(
                                pos in window for pos in ["increase", "grow", "rose", "rising", "climb", "upward"]
                            ):
                                warnings.append(
                                    f"Validation Warning: Trend contradiction for '{col}' in '{fname}'. "
                                    f"Data=UP but body has negative language."
                                )
                        elif direction == "down":
                            positives = ["increase", "grow", "rose", "upward", "climb", "expansion"]
                            if any(pos in window for pos in positives) and not any(
                                neg in window for neg in ["decrease", "decline", "drop", "fell", "downward", "shrank", "reduced"]
                            ):
                                warnings.append(
                                    f"Validation Warning: Trend contradiction for '{col}' in '{fname}'. "
                                    f"Data=DOWN but body has positive language."
                                )
                        idx += len(clean_col)
 
    if not has_files and doc_type in (
        "business_report", "sop", "policy_document", "financial_report",
        "circular", "office_memorandum", "file_note", "purchase_note", "office_order"
    ):
        patterns = [
            (r'\b\d+(?:\.\d+)?%', "percentage metric"),
            (r'₹\s*\d+', "Rupee currency"),
            (r'Rs\.?\s*\d+', "Rupee currency"),
            (r'\b\d{1,2}/\d{1,2}/\d{4}\b', "date"),
        ]
        found_metrics = []
        for pattern, label in patterns:
            for m in set(re.findall(pattern, body)):
                found_metrics.append(f"'{m}' ({label})")
        if found_metrics:
            warnings.append(
                f"Validation Warning: Potential invented metrics in no-file generation: {', '.join(found_metrics)}."
            )
 
    return warnings
 
 
# ==============================================================================
# INSIGHT VALIDATION  ← NEW
# Lightweight inline check: does each digest bullet have a data reference?
# ==============================================================================
 
def _validate_digest(digest_text: str, file_name: str) -> list[str]:
    """
    Check each bullet in a digest for unsupported claims.
    A bullet is flagged if it makes a numeric claim but has no
    supporting_data_ref marker AND no parenthetical source hint.
 
    This is the lightweight version of structured output validation —
    no extra LLM call, purely pattern-based.
 
    Returns list of warning strings (empty = all good).
    """
    warnings = []
    lines = [l.strip() for l in digest_text.splitlines() if l.strip().startswith("- ")]
 
    for line in lines:
        has_number = bool(re.search(r'\b\d+[\.,]?\d*\b', line))
        # A data ref is present if line contains parenthetical source hint
        # e.g. "(Sheet2)", "(Row 14)", "(Q3)", "(col: Sales KL)"
        has_ref = bool(re.search(
            r'\((?:sheet|row|col|table|data|file|period|Q\d|FY\d|avg|sum|total)',
            line, re.IGNORECASE
        ))
        # Also accept explicit field name references (>4 chars) matching known patterns
        has_field_ref = bool(re.search(
            r'\b(?:total|average|avg|mean|sum|max|min|rate|ratio|count|kl|mt|inr|lakh)\b',
            line, re.IGNORECASE
        ))
 
        if has_number and not has_ref and not has_field_ref:
            warnings.append(
                f"[DigestCheck] Unsupported numeric claim in {file_name}: {line[:120]}"
            )
 
    return warnings
 
 
# ==============================================================================
# SELF-CORRECTION  ← NEW
# One retry pass when severe validation issues are detected.
# ==============================================================================
 
def _self_correct_body(
    body: str,
    warnings: list[str],
    prompt: str,
    output_format: str,
    doc_type: str,
    official_format_spec: str | None,
    digest_results: list,
    cross_file_context: str,
) -> str:
    """
    Attempt one self-correction LLM call when severe structural warnings
    are found in the generated body (missing mandatory sections, template
    leftovers in purchase notes, or zero cross-file connections).
 
    Only fires for severe warnings — not for minor style notes.
    Returns corrected body, or original body if correction fails.
    """
    from llm_agent import query_llama
 
    SEVERE_PATTERNS = [
        "missing mandatory",
        "template instruction text",
        "too short",
        "missing 'Subject:'",
        "missing signatory",
        "None of the prioritized cross-file insights",
    ]
 
    severe = [w for w in warnings if any(p in w for p in SEVERE_PATTERNS)]
    if not severe:
        return body  # nothing severe — skip retry
 
    logger.warning(
        f"[Orchestrator] Self-correction triggered: {len(severe)} severe warning(s). "
        f"Retrying generation..."
    )
 
    issues_block = "\n".join(f"  - {w}" for w in severe)
 
    correction_prefix = (
        f"SELF-CORRECTION REQUIRED:\n"
        f"Your previous output had the following structural issues that MUST be fixed:\n"
        f"{issues_block}\n\n"
        f"Re-generate the complete document now, fixing ALL of the above issues.\n"
        f"Do NOT repeat the same mistakes.\n\n"
    )
 
    try:
        corrected_raw, _ = run_insight(
            digest_results,
            correction_prefix + prompt,
            output_format=output_format,
            doc_type=doc_type,
            cross_file_context=cross_file_context,
            official_format_spec=official_format_spec,
        )
        logger.info("[Orchestrator] Self-correction LLM call completed.")
        return corrected_raw
    except Exception as e:
        logger.warning(f"[Orchestrator] Self-correction failed: {e}. Using original body.")
        return body
 
 
def run_agent_from_api(prompt, file_paths=None, file_path=None,
                       output_format="docx", has_files=None,
                       add_acknowledgement=False, warnings_out=None):
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
        warnings_out        : optional list to collect validation warnings
 
    Returns:
        For pptx  -> (output_path, slide_count)
        For others -> output_path
    """
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/charts", exist_ok=True)
 
    # -- Normalise inputs -------------------------------------------------------
    if file_paths is None:
        file_paths = []
    if file_path and file_path not in file_paths:
        file_paths.append(file_path)
    file_paths = [fp for fp in file_paths if fp and os.path.exists(fp)]
 
    if has_files is None:
        has_files = len(file_paths) > 0
 
    logger.info(f"Files received: {len(file_paths)} -> "
                f"{[os.path.basename(f) for f in file_paths]}")
    logger.info(f"has_files={has_files} | output_format={output_format} | "
                f"add_acknowledgement={add_acknowledgement}")
 
    # -- Detect document type from prompt --------------------------------------
    doc_type = _detect_doc_type_from_prompt(prompt)
    logger.info(f"Prompt-detected doc_type: {doc_type}")
 
    # -- Resolve official format for government/PSU document types -------------
    official_format_spec = _resolve_official_format(doc_type, prompt)
    if official_format_spec:
        logger.info(f"[Orchestrator] Official format resolved for doc_type='{doc_type}'")
 
    seen_signatures = set()
    ordered_charts  = []
 
    # ==========================================================================
    # STEP 1 -- Extract data from all files
    # ==========================================================================
    file_data: dict = {}
 
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
                logger.info(f"    sheet={sname} | rows={sdata.get('row_count','')} | "
                            f"candidates={len(sdata.get('chart_candidates', []))}")
 
    # ==========================================================================
    # STEP 1b -- Detect intent per file
    # ==========================================================================
    file_intents:     dict[str, str]   = {}
    file_confidences: dict[str, float] = {}
 
    for name, data in file_data.items():
        intent, confidence = detect_content_intent(data, user_prompt=prompt)
        file_intents[name]     = intent
        file_confidences[name] = confidence
        data['intent']         = intent
        data['confidence']     = confidence
 
        if intent == "analytical":
            data["dataset_role"] = "Primary Metrics Registry"
        elif intent == "operational":
            data["dataset_role"] = "Operational Transaction Log"
        elif intent == "policy":
            data["dataset_role"] = "Policy Reference Document"
        elif intent == "educational":
            data["dataset_role"] = "Training Reference Manual"
        else:
            data["dataset_role"] = "Background Information Reference"
 
        logger.info(f"Intent for {name}: {intent} (confidence={confidence}, role={data['dataset_role']})")
 
    dominant_intent = (
        Counter(file_intents.values()).most_common(1)[0][0]
        if file_intents else "informational"
    )
    logger.info(f"Dominant intent: {dominant_intent}")
 
    # ==========================================================================
    # STEP 1c -- FIX: Use planner to lock doc_type when not set from prompt
    # Previously this was a simple dict lookup. Now we use plan_tasks() so the
    # planner's intent-routing logic, official-doc detection, and output format
    # resolution all happen in one place and feed back into the orchestrator.
    # ==========================================================================
    if doc_type is None and file_data:
        intent_list = list(file_intents.values())
        plan = plan_tasks(
            prompt       = prompt,
            file_names   = list(file_data.keys()),
            file_intents = intent_list,
        )
        # Only accept doc_type from planner if it resolved something non-trivial
        planner_doc_type = plan.get("doc_type")
        if planner_doc_type and planner_doc_type != "informational":
            doc_type = planner_doc_type
            logger.info(f"[Orchestrator] doc_type locked by planner: {doc_type}")
        else:
            # Fallback: simple intent map (keeps old behaviour for informational files)
            _intent_to_dtype = {
                "analytical":    "business_report",
                "operational":   "operational",
                "educational":   "educational",
                "policy":        "policy_document",
                "informational": None,
            }
            doc_type = _intent_to_dtype.get(dominant_intent)
            if doc_type:
                logger.info(f"[Orchestrator] doc_type inferred from intent: {doc_type}")
 
    elif doc_type is None and not file_data:
        # No files, no prompt signal — default
        doc_type = None
 
    # ==========================================================================
    # STEP 1d -- DataFusionAgent
    # ==========================================================================
    cross_file_context  = ""
    ranked_file_names: list[str] = list(file_data.keys())
    cross_file_insights = []
 
    if has_files and file_data:
        logger.info("[Orchestrator] Running DataFusionAgent ...")
        try:
            fusion_agent  = DataFusionAgent(file_data, user_prompt=prompt)
            fusion_result = fusion_agent.run()
            cross_file_insights = fusion_result.insights
 
            cross_file_context = fusion_result.cross_file_context
            logger.info(f"[Orchestrator] Fusion summary: {fusion_result.summary_log}")
 
            if fusion_result.ranked_files:
                ranked_file_names = [
                    f for f, _ in fusion_result.ranked_files if f in file_data
                ]
                logger.info("[Orchestrator] File relevance ranking:")
                for fname, score in fusion_result.ranked_files:
                    logger.info(f"  [{score:.2f}] {fname}")
 
            critical_high = [
                i for i in fusion_result.insights
                if i.priority in ("Critical", "High")
            ]
            if critical_high:
                logger.info(
                    f"[Orchestrator] {len(critical_high)} Critical/High "
                    f"cross-file insights — injecting into report prompt."
                )
 
        except Exception as fusion_err:
            logger.warning(
                f"[Orchestrator] DataFusionAgent failed (non-fatal): {fusion_err}"
            )
            cross_file_context = ""
 
    # ==========================================================================
    # STEP 2 -- Pre-compute deterministic data-charts per file
    # ==========================================================================
    data_charts_by_file: dict = {}
    total_data_charts = 0
 
    if doc_type not in _OFFICIAL_DOC_TYPES:
        for name, data in file_data.items():
            if file_intents.get(name, "analytical") in ("analytical", "operational"):
                charts = generate_charts_from_data(data)
            else:
                charts = []
            data_charts_by_file[name] = charts
            total_data_charts += len(charts)
            logger.info(f"Data charts from {name}: {len(charts)}")
        logger.info(f"Total data charts: {total_data_charts}")
    else:
        logger.info(f"[Orchestrator] Skipping chart generation for official doc type '{doc_type}'")
 
    # ==========================================================================
    # STEP 3 -- Generate title
    # ==========================================================================
    title = _make_title(
        prompt, list(file_data.keys()), dominant_intent, file_data, doc_type=doc_type
    )
    time.sleep(LLM_CALL_DELAY)
 
    # ==========================================================================
    # STEP 4 -- Build document body
    # ==========================================================================
    digest_results = []   # populated in has_files path; needed for self-correction
 
    if not has_files:
        from llm_agent import query_llama
        logger.info("No files — pure LLM generation (noinput mode)")
 
        augmented_prompt = prompt
        if official_format_spec:
            augmented_prompt = prompt + official_format_spec
            logger.info("[Orchestrator] Injecting official format spec into noinput prompt.")
 
        raw_body = query_llama(
            augmented_prompt,
            output_type=output_format,
            has_files=False,
            doc_type=doc_type
        )
 
        if doc_type not in _OFFICIAL_DOC_TYPES:
            body, body_charts = generate_charts_from_markers(raw_body)
            deduped, seen_signatures = dedup_charts(body_charts, seen_signatures)
            ordered_charts.extend(deduped)
        else:
            body = raw_body
 
        # FIX: pass has_files=False so no-file hallucination checks run correctly
        validation_warnings = _validate_body(
            body, file_data, [], title=title,
            doc_type=doc_type, prompt=prompt, has_files=False
        )
        severe_warnings = [
            w for w in validation_warnings
            if "Potential invented" in w or "Potential fabricated" in w
        ]
 
        if severe_warnings:
            logger.warning(
                f"[Orchestrator] Hallucination detected in noinput pass: {severe_warnings}. Retrying..."
            )
            time.sleep(LLM_CALL_DELAY)
 
            retry_prompt = (
                augmented_prompt +
                "\n\nCRITICAL RETRY: Your previous generation contained invented "
                "metrics/dates/values: " + ", ".join(severe_warnings)[:200] +
                ". Use underlines (__________________) for ALL unknown fields. "
                "Never write specific numbers, dates, or names."
            )
            raw_body = query_llama(
                retry_prompt,
                output_type=output_format,
                has_files=False,
                doc_type=doc_type
            )
            if doc_type not in _OFFICIAL_DOC_TYPES:
                body, body_charts = generate_charts_from_markers(raw_body)
                ordered_charts = []
                deduped, seen_signatures = dedup_charts(body_charts, seen_signatures)
                ordered_charts.extend(deduped)
            else:
                body = raw_body
 
    else:
        # -- STEP 4a -- Per-file fact digests -----------------------------------
        import hashlib
        import json
        
        cache_dir = os.path.join("outputs", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_input = f"{prompt}:::{'|'.join(ranked_file_names)}"
        cache_hash = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
        cache_path = os.path.join(cache_dir, f"{cache_hash}.json")
        
        cache_data = {"digests": {}}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                logger.info(f"Loaded session cache from {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to read cache file {cache_path}: {e}")

        for name in ranked_file_names:
            if name in cache_data["digests"]:
                logger.info(f"Using cached digest for {name}")
                cached_entry = cache_data["digests"][name]
                clean_digest = cached_entry["digest_text"]
                detected_intent = cached_entry["detected_intent"]
                cached_charts = cached_entry.get("charts", [])
                
                digest_results.append((clean_digest, detected_intent))
                
                if doc_type not in _OFFICIAL_DOC_TYPES:
                    processed_charts = []
                    for item in cached_charts:
                        title_c = item[0]
                        path_c = item[1]
                        sig_c = item[2]
                        col_c = item[3] if len(item) > 3 else ""
                        processed_charts.append((title_c, path_c, tuple(sig_c) if sig_c else None, col_c))
                    
                    deduped, seen_signatures = dedup_charts(processed_charts, seen_signatures)
                    ordered_charts.extend(deduped)
                continue

            data         = file_data[name]
            dtype        = data.get('type', 'unknown')
            intent       = file_intents.get(name, "analytical")
            dataset_role = data.get("dataset_role", "Background Information Reference")
            causal_hints = data.get("causal_hints", [])
 
            logger.info(f"Digesting [{dtype}] (intent={intent}, role={dataset_role}): {name}")
 
            prior_digests_summary = ""
            if len(digest_results) > 0:
                prior_digests_summary = "\n\n".join(
                    f"--- Findings from {ranked_file_names[i]} ---\n{digest_results[i][0]}"
                    for i in range(len(digest_results))
                )
 
            if dtype == 'multi_sheet':
                flat = _flatten_multisheet(data)
                flat["dataset_role"] = dataset_role
                flat["causal_hints"] = causal_hints
                digest_text, detected_intent = run_analysis(
                    flat,
                    instruction=f"{prompt} — analysing {_display_name(name)}",
                    intent=intent,
                    dataset_role=dataset_role,
                    causal_hints=causal_hints,
                    prior_digests_summary=prior_digests_summary
                )
            else:
                digest_text, detected_intent = run_analysis(
                    data,
                    instruction=f"{prompt} — analysing {_display_name(name)}",
                    intent=intent,
                    dataset_role=dataset_role,
                    causal_hints=causal_hints,
                    prior_digests_summary=prior_digests_summary
                )
 
            # FIX: digest validation — flag unsupported numeric claims
            digest_warnings = _validate_digest(digest_text, name)
            if digest_warnings:
                for dw in digest_warnings:
                    logger.warning(dw)
                    if isinstance(warnings_out, list):
                        warnings_out.append(dw)
 
            if doc_type not in _OFFICIAL_DOC_TYPES:
                clean_digest, marker_charts = generate_charts_from_markers(digest_text)
 
                if intent in ("analytical", "operational"):
                    file_charts = data_charts_by_file.get(name, []) + marker_charts
                else:
                    file_charts = marker_charts
 
                deduped, seen_signatures = dedup_charts(file_charts, seen_signatures)
                ordered_charts.extend(deduped)
                logger.info(
                    f"Charts kept for {name}: {len(deduped)} "
                    f"(data={len(data_charts_by_file.get(name, []))}, "
                    f"markers={len(marker_charts)})"
                )
                digest_results.append((clean_digest, detected_intent))

                serializable_charts = []
                for title_c, path_c, sig_c, col_c in file_charts:
                    serializable_charts.append([title_c, path_c, list(sig_c) if sig_c else None, col_c])
                
                cache_data["digests"][name] = {
                    "digest_text": clean_digest,
                    "detected_intent": detected_intent,
                    "charts": serializable_charts
                }
            else:
                digest_results.append((digest_text, detected_intent))
                cache_data["digests"][name] = {
                    "digest_text": digest_text,
                    "detected_intent": detected_intent,
                    "charts": []
                }

            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, indent=2, default=str)
                logger.info(f"Saved session cache update for {name} to {cache_path}")
            except Exception as cache_err:
                logger.warning(f"Failed to save cache: {cache_err}")
 
            time.sleep(LLM_CALL_DELAY)
 
        # -- STEP 4b -- Single integrated report from all digests ---------------
        logger.info(
            f"Generating integrated {output_format} report from "
            f"{len(digest_results)} digest(s) "
            f"(intent={dominant_intent}, doc_type={doc_type}) ..."
        )
 
        raw_report, _intent = run_insight(
            digest_results,
            prompt,
            output_format=output_format,
            doc_type=doc_type,
            cross_file_context=cross_file_context,
            official_format_spec=official_format_spec
        )
 
        if doc_type not in _OFFICIAL_DOC_TYPES:
            clean_report, report_charts = generate_charts_from_markers(raw_report)
            deduped, seen_signatures    = dedup_charts(report_charts, seen_signatures)
            ordered_charts.extend(deduped)
            logger.info(
                f"Charts kept for integrated report: {len(deduped)} "
                f"(of {len(report_charts)})"
            )
            body = clean_report
        else:
            body = raw_report
 
    # -- STEP 4c -- Output validation ------------------------------------------
    validation_warnings = _validate_body(
        body, file_data, cross_file_insights,
        title=title, doc_type=doc_type, prompt=prompt, has_files=has_files
    )
    if validation_warnings:
        for w in validation_warnings:
            logger.warning(w)
            if isinstance(warnings_out, list):
                warnings_out.append(w)
    else:
        logger.info("[Orchestrator] Output validation passed.")
 
    # -- STEP 4d -- Self-correction (has_files path only) ← NEW ---------------
    if has_files and validation_warnings and digest_results:
        corrected = _self_correct_body(
            body                 = body,
            warnings             = validation_warnings,
            prompt               = prompt,
            output_format        = output_format,
            doc_type             = doc_type,
            official_format_spec = official_format_spec,
            digest_results       = digest_results,
            cross_file_context   = cross_file_context,
        )
        if corrected != body:
            body = corrected
            # Re-validate after correction so final warnings reflect corrected output
            post_warnings = _validate_body(
                body, file_data, cross_file_insights,
                title=title, doc_type=doc_type, prompt=prompt, has_files=has_files
            )
            remaining = len(post_warnings)
            fixed     = len(validation_warnings) - remaining
            logger.info(
                f"[Orchestrator] Self-correction result: "
                f"{fixed} warning(s) resolved, {remaining} remaining."
            )
            if isinstance(warnings_out, list):
                # Replace old warnings with post-correction set
                warnings_out.clear()
                warnings_out.extend(post_warnings)
 
    # ==========================================================================
    # STEP 5 -- Verify chart paths
    # ==========================================================================
    def select_top_high_value_charts(charts, max_charts=6):
        categories = {
            "revenue_trend":       ["revenue", "turnover trend"],
            "market_comparison":   ["market", "comparison", "hpcl vs", "omc", "bpcl", "iocl", "reliance", "nayara"],
            "inventory_comparison":["inventory", "closing stock", "opening stock", "stock levels"],
            "turnover_comparison": ["turnover ratio", "turnover comparison", "stock movement"],
            "stock_imbalance":     ["imbalance", "stock share", "sales share", "stock vs sales"],
            "product_contribution":["contribution", "product-wise", "product sales", "sales by product"]
        }
        selected  = {}
        remaining = []
        for chart in charts:
            title_c, path, sig = chart
            title_lower = title_c.lower()
            matched_cat = None
            for cat, keywords in categories.items():
                if any(k in title_lower for k in keywords):
                    matched_cat = cat
                    break
            if matched_cat:
                if matched_cat not in selected:
                    selected[matched_cat] = chart
                else:
                    remaining.append(chart)
            else:
                remaining.append(chart)
        result = list(selected.values())
        for r in remaining:
            if len(result) >= max_charts:
                break
            result.append(r)
        return result
 
    ordered_charts = select_top_high_value_charts(ordered_charts, max_charts=6)
    logger.info(f"Total charts to embed: {len(ordered_charts)}")
    verified = [(h, p, col) for h, p, _sig, col in ordered_charts if os.path.exists(p)]
    logger.info(f"Verified chart paths: {len(verified)}")
 
    # ==========================================================================
    # STEP 6 -- Write document
    # ==========================================================================
    output_path = _resolve_path(_sanitize(title, output_format))

    # Clean up session cache on success
    if 'cache_path' in locals() and os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Cleaned up session cache: {cache_path}")
        except Exception as cache_cleanup_err:
            logger.warning(f"Failed to delete session cache {cache_path}: {cache_cleanup_err}")

    if output_format == "docx":
        generate_docx(body, output_path, title, chart_images=verified,
                      doc_type=doc_type, user_prompt=prompt)
        return output_path
 
    elif output_format == "pdf":
        generate_pdf(body, output_path, title, chart_images=verified,
                     doc_type=doc_type, user_prompt=prompt)
        return output_path
 
    elif output_format == "pptx":
        slide_count = generate_ppt(
            body, output_path, filename_title=title,
            chart_images=verified, add_acknowledgement=add_acknowledgement
        )
        logger.info(f"PPT saved: {output_path} ({slide_count} slides)")
        return output_path, slide_count
 
    else:
        raise ValueError(f"Unsupported format: {output_format}")
 
 
# ==============================================================================
# Helpers
# ==============================================================================
 
def _flatten_multisheet(data: dict) -> dict:
    sheets               = data.get('sheets', {})
    all_candidates       = []
    all_samples          = {}
    all_stats            = {}
    all_columns          = []
    all_entity_stats     = []
    all_trends           = {}
    all_domain_map       = {}
    all_causal_hints     = []
    all_contradiction_seeds = []
 
    for sheet_name, sdata in sheets.items():
        all_candidates.extend(sdata.get('chart_candidates', []))
        all_columns.extend(sdata.get('columns', []))
        if sdata.get('sample'):
            all_samples[sheet_name] = sdata['sample']
        if sdata.get('stats'):
            all_stats[sheet_name]   = sdata['stats']
        if sdata.get('entity_stats'):
            for est in sdata['entity_stats']:
                est_copy = est.copy()
                est_copy['sheet'] = sheet_name
                all_entity_stats.append(est_copy)
        if sdata.get('trends'):
            for col, t in sdata['trends'].items():
                all_trends[f"{sheet_name}_{col}"] = t
        if sdata.get('domain_map'):
            for col, dom in sdata['domain_map'].items():
                all_domain_map[f"{sheet_name}_{col}"] = dom
        if sdata.get('causal_hints'):
            all_causal_hints.extend(sdata['causal_hints'])
        if sdata.get('contradiction_seeds'):
            all_contradiction_seeds.extend(sdata['contradiction_seeds'])
 
    return {
        "file":                 data.get('file', ''),
        "type":                 "tabular",
        "columns":              list(dict.fromkeys(all_columns)),
        "sheets_summary":       {
            s: {"rows": d.get('row_count'), "columns": d.get('columns')}
            for s, d in sheets.items()
        },
        "sample":               all_samples,
        "stats":                all_stats,
        "chart_candidates":     all_candidates[:16],
        "entities":             data.get('entities', {}),
        "cross_sheet_entities": data.get('cross_sheet_entities', {}),
        "entity_stats":         all_entity_stats,
        "trends":               all_trends,
        "domain_map":           all_domain_map,
        "causal_hints":         list(dict.fromkeys(all_causal_hints)),
        "contradiction_seeds":  list(dict.fromkeys(all_contradiction_seeds)),
    }
 
 
# -- Title generation ----------------------------------------------------------
 
_TITLE_FILLER = {
    'give', 'me', 'a', 'an', 'the', 'make', 'create', 'generate',
    'write', 'build', 'produce', 'get', 'please', 'can', 'you',
    'i', 'want', 'need', 'help', 'with', 'for', 'my', 'our',
}
 
_FORCE_UPPER = {
    'hpcl', 'bpcl', 'iocl', 'lpg', 'atf', 'kl', 'mt',
    'omc', 'hr', 'fy', 'kpi', 'pdf', 'ppt', 'ai', 'api',
}
 
 
def _title_case(text: str) -> str:
    minor  = {'and', 'or', 'the', 'of', 'in', 'on', 'with', 'a',
              'an', 'to', 'for', 'at', 'by', 'from', 'into', 'as'}
    words  = text.strip().split()
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
    strip_words = {
        'generate', 'create', 'write', 'draft', 'make', 'produce', 'act', 'as', 'senior',
        'management', 'consultant', 'prepare', 'a', 'an', 'the', 'for', 'me', 'please',
        'give', 'doc', 'report', 'document', 'file', 'note', 'notice', 'memorandum',
        'circular', 'order', 'memo', 'docx', 'pdf', 'pptx', 'in', 'format', 'on', 'with',
        'and', 'about', 'our', 'us', 'office', 'purchase', 'procurement', 'corporate',
        'internal', 'official', 'departmental', 'department', 'of', 'to', 'from', 'at',
        'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'will', 'be',
        'have', 'has', 'had', 'do', 'does', 'did', 'some', 'any', 'all', 'new', 'old',
        'business'
    }
    clean_words = [
        w for w in re.findall(r'\b\w+\b', prompt.lower())
        if w not in strip_words and len(w) > 2
    ]
    subject = " ".join(clean_words[:5]) or "HPCL Operations"
    return _title_case(subject)
 
 
def _make_title(prompt: str, file_names: list,
                intent: str = "informational",
                file_data: dict = None,
                doc_type: str = None) -> str:
    from llm_agent import query_llama
 
    intent_hint = {
        "analytical":    "analytical/KPI report",
        "informational": "informational document",
        "educational":   "educational or training document",
        "policy":        "policy or compliance document",
        "operational":   "operational status report",
    }.get(intent, "document")
 
    content_lines = []
    if file_data:
        for fname, data in file_data.items():
            clean = _display_name(fname)
            if data.get("type") == "multi_sheet":
                cols = []
                for s in data.get("sheets", {}).values():
                    cols.extend(s.get("columns", []))
            else:
                cols = data.get("columns", [])
            text_preview = ""
            if data.get("type") == "text":
                text_preview = data.get("content", "")[:300]
            if cols:
                content_lines.append(f"File: {clean} | Columns: {', '.join(cols[:12])}")
            elif text_preview:
                content_lines.append(f"File: {clean} | Content: {text_preview}")
            else:
                content_lines.append(f"File: {clean}")
 
    content_context = "\n".join(content_lines)
 
    llm_prompt = (
        f"You are a professional title generator for HPCL corporate documents.\n"
        f"Generate a professional, concise title (5-8 words) based on the user's request and files.\n\n"
        f"User Request: {prompt[:300]}\n"
        f"Document Type: {doc_type or intent_hint}\n"
    )
    if content_context:
        llm_prompt += f"Uploaded file details:\n{content_context}\n\n"
    llm_prompt += (
        "STRICT Rules:\n"
        "- Title must directly reflect the document subject.\n"
        "- NEVER include prompt instructions like 'Act As', 'Senior Consultant', 'Prepare', 'Draft', 'Create'.\n"
        "- Reply with ONLY the final title. No quotes, no markdown, no punctuation at end."
    )
 
    try:
        raw   = query_llama(llm_prompt, output_type="plan", has_files=bool(file_names))
        title = raw.strip().strip('"\'').strip('.')
        title = re.sub(r'^#+\s*', '', title)
        title = re.sub(r'^(title|document title):\s*', '', title, flags=re.IGNORECASE)
        title = title.strip().strip('"\'').strip('.')
        title = _title_case(title)
 
        forbidden_fragments = ["act as", "management consultant", "prepare a", "draft a", "generate a"]
        if any(f in title.lower() for f in forbidden_fragments) or len(title) < 5:
            raise ValueError("LLM title had forbidden fragments or was too short.")
 
        for key in ["file note", "office memorandum", "office notice", "circular",
                    "purchase note", "office order"]:
            title = re.sub(rf'^{key}\s*:?\s*', '', title, flags=re.IGNORECASE)
        title = title.strip()
 
        if doc_type:
            doc_title_map = {
                "file_note":         "File Note: ",
                "office_memorandum": "Office Memorandum: ",
                "office_notice":     "Office Notice: ",
                "circular":          "Circular: ",
                "purchase_note":     "Purchase Note: ",
                "office_order":      "Office Order: "
            }
            if doc_type in doc_title_map:
                title = f"{doc_title_map[doc_type]}{title}"
 
        return title
 
    except Exception as e:
        logger.warning(f"Title LLM call failed ({e}), using fallback.")
        prefix = ""
        if doc_type:
            doc_title_map = {
                "file_note":         "File Note: ",
                "office_memorandum": "Office Memorandum: ",
                "office_notice":     "Office Notice: ",
                "circular":          "Circular: ",
                "purchase_note":     "Purchase Note: ",
                "office_order":      "Office Order: "
            }
            prefix = doc_title_map.get(doc_type, f"{doc_type.replace('_', ' ').title()}: ")
        return f"{prefix}{_fallback_title(prompt)}"
 
 
def _sanitize(text: str, ext: str) -> str:
    base = re.sub(r'[^\w\s-]', '', text).strip().lower()
    base = re.sub(r'[\s-]+', '_', base) or uuid.uuid4().hex[:8]
    return f"outputs/{base}.{ext}"
 
 
def _resolve_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while os.path.exists(f"{base}({i}){ext}"):
        i += 1
    return f"{base}({i}){ext}"
 
 
def run_agent_pipeline():
    print("\n=== HPCL AI Document Agent CLI ===")
    prompt = input("Enter your prompt / document requirements:\n>>> ").strip()
    if not prompt:
        print("Prompt cannot be empty.")
        return
 
    file_input = input("Enter paths of files to attach (comma-separated, optional):\n>>> ").strip()
    file_paths = []
    if file_input:
        file_paths = [fp.strip().strip('"\'') for fp in file_input.split(',')]
        file_paths = [fp for fp in file_paths if fp]
 
    output_format = input("Enter output format (docx, pdf, pptx) [default: docx]:\n>>> ").strip().lower()
    if not output_format:
        output_format = "docx"
    elif output_format not in ("docx", "pdf", "pptx"):
        print(f"Unsupported format '{output_format}'. Defaulting to docx.")
        output_format = "docx"
 
    print("\nRunning pipeline, please wait...")
    try:
        result = run_agent_from_api(
            prompt=prompt,
            file_paths=file_paths,
            output_format=output_format
        )
        if isinstance(result, tuple):
            output_path, slide_count = result
            print(f"\n✅ Success! Generated presentation: {output_path} ({slide_count} slides)")
        else:
            print(f"\n✅ Generated document: {result}")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")