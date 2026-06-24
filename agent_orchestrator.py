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
        # "notice" alone is intentionally excluded -- too broad.
        # Matches "notice period policy", "notice board" etc. incorrectly.
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
 
# Set of official document types that require format resolution
_OFFICIAL_DOC_TYPES = {
    "file_note", "office_memorandum", "office_notice",
    "circular", "purchase_note", "office_order"
}

# High-priority document types where explicit request should not be overridden by generic topic keywords
_HIGH_PRIORITY_DOC_TYPES = {
    "file_note", "office_memorandum", "office_notice",
    "circular", "purchase_note", "office_order",
    "business_report", "financial_report", "sop"
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
        # Only apply pluralization logic to the last word
        if i == len(words) - 1 and w[-1].isalnum():
            w_lower = w.lower()
            if w_lower.endswith('y'):
                # policy -> policies
                base = re.escape(w[:-1])
                escaped_words.append(f"(?:{base}y|{base}ies)")
            elif w_lower.endswith('um'):
                # memorandum -> memorandums / memoranda
                base = re.escape(w[:-2])
                escaped_words.append(f"(?:{base}um|{base}ums|{base}a)")
            elif any(w_lower.endswith(sfx) for sfx in ('s', 'x', 'z', 'ch', 'sh')):
                escaped_words.append(f"{re.escape(w)}(?:es)?")
            else:
                escaped_words.append(f"{re.escape(w)}s?")
        else:
            escaped_words.append(re.escape(w))
            
    escaped_sig = r'\s+'.join(escaped_words)
    
    # Construct regex pattern
    pattern = ""
    # Start boundary: if first char is alphanumeric, enforce word boundary
    if sig_stripped[0].isalnum():
        pattern += r'\b'
    pattern += escaped_sig
    # End boundary: if last char is alphanumeric, enforce word boundary
    if sig_stripped[-1].isalnum():
        pattern += r'\b'
        
    # Compile and search
    return bool(re.search(pattern, pl))


def _detect_doc_type_from_prompt(prompt: str) -> str | None:
    pl = prompt.lower()
    matches = [] # list of (doc_type, match_length, is_high_priority)
    for doc_type, signals in _DOCTYPE_SIGNALS.items():
        for sig in signals:
            if _match_signal(pl, sig):
                is_high = doc_type in _HIGH_PRIORITY_DOC_TYPES
                matches.append((doc_type, len(sig.strip()), is_high))
    if matches:
        # Sort by is_high_priority (True first), then by match length descending
        matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
        detected = matches[0][0]
        logger.info(f"Doc type detected from prompt: {detected} (matches: {matches})")
        return detected
    return None
 
 
def _resolve_official_format(doc_type: str, subject: str) -> str | None:
    """
    For official government/PSU document types, retrieves the standard format/structure
    defined in the document rules from llm_agent.py.
 
    Returns a format instruction string to inject into the generation prompt,
    or None if doc_type is not an official type.
    """
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
 
 
def _validate_body(body: str, file_data: dict, cross_file_insights: list = None,
                   title: str = None, doc_type: str = None, prompt: str = "") -> list[str]:
    warnings = []
 
    if not body or len(body.strip()) < 200:
        warnings.append("Validation Warning: Generated body is too short (< 200 chars). Provide more detailed instructions.")
 
    # For official docs, don't require ## headings in the same way
    if doc_type not in _OFFICIAL_DOC_TYPES and '##' not in body:
        warnings.append("Validation Warning: No section headings (##) found in output. Structure the document with appropriate headers.")
 
    # 1. Title generic validation
    generic_titles = ["hpcl document", "untitled document", "report", "learning module",
                      "employee training program", "development program", "kpi dashboard",
                      "training presentation"]
    if title and any(gt == title.lower().strip() for gt in generic_titles):
        warnings.append(f"Validation Warning: Document title '{title}' appears generic. Generate a specific, descriptive title.")
 
    # 2. Section placeholders check -- skip for official docs (they use underlines intentionally)
    if doc_type not in _OFFICIAL_DOC_TYPES:
        placeholders = ['[insert', '[tbd]', '[todo]', '[placeholder]', 'lorem ipsum']
        for ph in placeholders:
            if ph in body.lower():
                warnings.append(f"Validation Warning: Placeholder text '{ph}' found. Replace all placeholders with actual content.")
 
    # 3. File reference check -- only for non-official docs
    if doc_type not in _OFFICIAL_DOC_TYPES and file_data:
        for fname, data in file_data.items():
            dname = _display_name(fname).lower()
            cols  = []
            if data.get('type') == 'multi_sheet':
                for s in data.get('sheets', {}).values():
                    cols.extend(s.get('columns', []))
            else:
                cols = data.get('columns', [])
 
            top_cols = [c.lower() for c in cols[:5]]
            found = (dname in body.lower() or
                     any(c in body.lower() for c in top_cols if len(c) > 4))
            if not found:
                warnings.append(f"Validation Warning: File '{fname}' is not referenced in the output. Integrate details from this file.")
 
    # 4. Multi-file comparison validation
    if len(file_data) > 1 and doc_type not in _OFFICIAL_DOC_TYPES:
        connecting_words = ["compare", "correlation", "contradict", "aligned", "alongside",
                            "compounded", "across", "versus", "vs", "difference"]
        has_connecting = any(w in body.lower() for w in connecting_words)
        if not has_connecting:
            warnings.append("Validation Warning: No cross-file comparison terms found. The report may be siloed across files.")
 
        if cross_file_insights:
            referenced_insights = 0
            for ins in cross_file_insights:
                words = re.findall(r'\b\w{4,}\b', ins.title.lower() + " " + ins.detail.lower())
                stop_words = {"file", "dataset", "report", "connection", "numeric", "variance",
                              "insight", "priority", "critical", "high", "medium", "low",
                              "value", "values", "common", "shared", "trends", "trend"}
                keywords = [w for w in words if w not in stop_words]
 
                if keywords:
                    # FIX: lowered threshold from min(2,...) to min(1,...) to reduce false warnings
                    hits = sum(1 for kw in keywords if kw in body.lower())
                    if hits >= min(1, len(keywords)):
                        referenced_insights += 1
 
            if referenced_insights == 0:
                warnings.append("Validation Warning: None of the prioritized cross-file insights were detected in the generated body. Ensure cross-file connections are analyzed.")
 
    # 5. Recommendation specificity check
    recs_match = re.search(r'##\s*Recommendations\b.*', body, re.IGNORECASE | re.DOTALL)
    if recs_match:
        recs_text = recs_match.group()
        has_numbers = any(char.isdigit() for char in recs_text)
        if not has_numbers:
            warnings.append(
                "Validation Warning: Recommendations section lacks numeric data or specific metrics. Tie recommendations to numeric findings."
            )
 
    # 5b. Recommendations structured sections check for business_report
    if doc_type == "business_report":
        recs_match = re.search(r'##\s*(?:Strategic\s+)?Recommendations\b.*', body, re.IGNORECASE | re.DOTALL)
        if recs_match:
            recs_text = recs_match.group()
            missing_structure = []
            for term in ["Recommendation", "Operational Rationale", "Expected Outcome & Metrics"]:
                if term.lower() not in recs_text.lower():
                    missing_structure.append(term)
            if missing_structure:
                warnings.append(
                    f"Validation Warning: Strategic recommendations are missing structured format sections: {missing_structure}. "
                    f"Each recommendation must include: Recommendation, Operational Rationale, and Expected Outcome & Metrics."
                )
 
    # 6. Official document structural compliance
    if doc_type in _OFFICIAL_DOC_TYPES:
        if "subject:" not in body.lower() and not any("subject:" in s.lower() for s in body.splitlines()[:15]):
            warnings.append("Validation Warning: Official document is missing the required 'Subject:' header block. Please ensure a subject line is included at the top.")
 
        sig_indicators = ["prepared by", "sd/-", "approved by", "recommended by", "submitted for approval"]
        if not any(ind in body.lower() for ind in sig_indicators):
            warnings.append("Validation Warning:\nNo approval authority or signatory block detected.")
 
        if doc_type in ["file_note", "office_memorandum", "circular"]:
            bullet_count = body.count("\n- ") + body.count("\n* ")
            if bullet_count > 5:
                warnings.append(
                    f"Validation Warning: Official document type '{doc_type}' should use numbered paragraphs "
                    f"rather than bullet points."
                )
 
        # Purchase Note specific template check
        if doc_type == "purchase_note":
            template_leftovers = [
                "write 3-5 numbered paragraphs",
                "explain what exactly",
                "provide a markdown table",
                "for unknown values",
                "write 2-3 numbered paragraphs",
                "write a clear, formal recommendation"
            ]
            found_leftovers = [tl for tl in template_leftovers if tl in body.lower()]
            if found_leftovers:
                warnings.append(
                    f"Validation Warning: Purchase Note contains template instruction text: {found_leftovers}. "
                    f"Ensure all template instructions are replaced with actual content."
                )
 
        # 6b. Mandatory sections check for official document types
        mandatory_checks = {
            "file_note": [
                ("## background / facts", "Validation Warning: File Note is missing the mandatory 'Background / Facts of the Case' section."),
                ("## analysis / deliberations", "Validation Warning: File Note is missing the mandatory 'Analysis / Deliberations' section."),
                ("## recommendation", "Validation Warning: File Note is missing the mandatory 'Recommendation' section."),
                ("## approval sought", "Validation Warning: File Note is missing the mandatory 'Approval Sought' section."),
                ("submitted for approval", "Validation Warning: File Note is missing the mandatory 'Submitted for Approval' signature block.")
            ],
            "office_memorandum": [
                ("no.:", "Validation Warning: Office Memorandum is missing 'No.:' in the header block."),
                ("from:", "Validation Warning: Office Memorandum is missing 'From:' in the header block."),
                ("to:", "Validation Warning: Office Memorandum is missing 'To:' in the header block."),
                ("## body", "Validation Warning: Office Memorandum is missing the mandatory 'Body' section."),
                ("sd/-", "Validation Warning: Office Memorandum is missing the 'Sd/-' signature marker.")
            ],
            "office_notice": [
                ("notice no.:", "Validation Warning: Office Notice is missing 'Notice No.:' in the header block."),
                ("## notice body", "Validation Warning: Office Notice is missing the mandatory 'Notice Body' section."),
                ("## compliance", "Validation Warning: Office Notice is missing the mandatory 'Compliance / Action Required' section."),
                ("by order", "Validation Warning: Office Notice is missing the 'By Order' authority line."),
                ("distribution:", "Validation Warning: Office Notice is missing the 'Distribution:' list.")
            ],
            "circular": [
                ("circular no.", "Validation Warning: Circular is missing 'Circular No.' in the header block."),
                ("## preamble", "Validation Warning: Circular is missing the mandatory 'Preamble / Background' section."),
                ("## instructions", "Validation Warning: Circular is missing the mandatory 'Instructions / Directives' section."),
                ("## effective date", "Validation Warning: Circular is missing the mandatory 'Effective Date' section."),
                ("distribution:", "Validation Warning: Circular is missing the 'Distribution:' list.")
            ],
            "purchase_note": [
                ("## 1. purpose", "Validation Warning: Purchase Note is missing 'Purpose and Operational Requirement' section."),
                ("## 2. specification", "Validation Warning: Purchase Note is missing 'Specification of Items / Services' section."),
                ("## 3. operational justification", "Validation Warning: Purchase Note is missing 'Operational Justification and Risk of Non-Procurement' section."),
                ("## 4. budget provision", "Validation Warning: Purchase Note is missing 'Budget Provision and Financial Considerations' section."),
                ("## 5. procurement method", "Validation Warning: Purchase Note is missing 'Procurement Method and Vendor Strategy' section."),
                ("## 6. proposed delivery", "Validation Warning: Purchase Note is missing 'Proposed Delivery Schedule' section."),
                ("## 7. recommendation", "Validation Warning: Purchase Note is missing 'Recommendation and Approval Sought' section.")
            ],
            "office_order": [
                ("office order no.", "Validation Warning: Office Order is missing 'Office Order No.' in the header block."),
                ("## order", "Validation Warning: Office Order is missing the mandatory 'Order' section."),
                ("## effective date", "Validation Warning: Office Order is missing the mandatory 'Effective Date' section."),
                ("## compliance", "Validation Warning: Office Order is missing the mandatory 'Compliance' section."),
                ("copy to:", "Validation Warning: Office Order is missing the 'Copy to:' distribution list.")
            ]
        }
        
        checks = mandatory_checks.get(doc_type, [])
        for term, warning_msg in checks:
            if term not in body.lower():
                warnings.append(warning_msg)
 
        # Anti-hallucination: check for invented person names
        honorific_match = re.search(
            r'\b(shri|smt|mr|ms|dr)\.?\s+[a-zA-Z]{2,}\s+[a-zA-Z]{2,}',
            body, re.IGNORECASE
        )
        if honorific_match:
            warnings.append(
                f"Validation Warning: Potential invented person name found: '{honorific_match.group()}'. "
                f"Use blank underlines for name fields to avoid hallucination."
            )
 
        # Check for invented specific dates (DD/MM/YYYY)
        date_patterns = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b', body)
        for dt in date_patterns:
            if dt not in prompt:
                in_files = any(dt in str(fdata) for fdata in file_data.values())
                if not in_files:
                    warnings.append(
                        f"Validation Warning: Potential invented date found: '{dt}'. "
                        f"Use blank underlines for unknown dates to avoid hallucination."
                    )
 
    # 7. Trend validation -- only for non-official analytical docs
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
                                    f"Data indicates UP but body has negative keywords."
                                )
                        elif direction == "down":
                            positives = ["increase", "grow", "rose", "upward", "climb", "expansion"]
                            if any(pos in window for pos in positives) and not any(
                                neg in window for neg in ["decrease", "decline", "drop", "fell", "downward", "shrank", "reduced"]
                            ):
                                warnings.append(
                                    f"Validation Warning: Trend contradiction for '{col}' in '{fname}'. "
                                    f"Data indicates DOWN but body has positive keywords."
                                )
                        idx += len(clean_col)
 
    return warnings
 
 
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
        has_files           : bool -- explicitly passed from app.py.
                              If None, inferred from file_paths.
        add_acknowledgement : bool -- add acknowledgement slide (pptx only)
 
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
    # STEP 1c -- Resolve effective doc_type from intent (if not set by prompt)
    # ==========================================================================
    if doc_type is None:
        _intent_to_dtype = {
            "analytical":    "business_report",
            "operational":   "operational",
            "educational":   "educational",
            "policy":        "policy_document",
            "informational": None,
        }
        doc_type = _intent_to_dtype.get(dominant_intent)
        if doc_type:
            logger.info(f"Doc type inferred from intent: {doc_type}")
 
    # ==========================================================================
    # STEP 1d -- DataFusionAgent
    # ==========================================================================
    cross_file_context = ""
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
                ranked_file_names = [f for f, _ in fusion_result.ranked_files
                                     if f in file_data]
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
                    f"cross-file insights -- injecting into report prompt."
                )
 
        except Exception as fusion_err:
            logger.warning(
                f"[Orchestrator] DataFusionAgent failed (non-fatal): {fusion_err}"
            )
            cross_file_context = ""
 
    # ==========================================================================
    # STEP 2 -- Pre-compute deterministic data-charts per file
    # (skip for official doc types -- they never have charts)
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
    title = _make_title(prompt, list(file_data.keys()), dominant_intent, file_data, doc_type=doc_type)
    time.sleep(LLM_CALL_DELAY)
 
    # ==========================================================================
    # STEP 4 -- Build document body
    # ==========================================================================
    if not has_files:
        from llm_agent import query_llama
        logger.info("No files -- pure LLM generation (noinput mode)")
 
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
 
    else:
        # -- STEP 4a -- Per-file fact digests -----------------------------------
        digest_results = []
 
        for name in ranked_file_names:
            data   = file_data[name]
            dtype  = data.get('type', 'unknown')
            intent = file_intents.get(name, "analytical")
            # FIX: fetch dataset_role from file_data (set in step 1b), not from local var
            dataset_role = file_data[name].get("dataset_role", "Background Information Reference")
            causal_hints = data.get("causal_hints", [])
            logger.info(f"Digesting [{dtype}] (intent={intent}, role={dataset_role}): {name}")
 
            if dtype == 'multi_sheet':
                flat = _flatten_multisheet(data)
                # FIX: carry dataset_role into the flattened dict
                flat["dataset_role"] = dataset_role
                flat["causal_hints"] = causal_hints
                digest_text, detected_intent = run_analysis(
                    flat,
                    instruction=f"{prompt} -- analysing {_display_name(name)}",
                    intent=intent,
                    dataset_role=dataset_role,
                    causal_hints=causal_hints
                )
            else:
                digest_text, detected_intent = run_analysis(
                    data,
                    instruction=f"{prompt} -- analysing {_display_name(name)}",
                    intent=intent,
                    dataset_role=dataset_role,
                    causal_hints=causal_hints
                )
 
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
            else:
                digest_results.append((digest_text, detected_intent))
 
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
        title=title, doc_type=doc_type, prompt=prompt
    )
    if validation_warnings:
        for w in validation_warnings:
            logger.warning(w)
            if isinstance(warnings_out, list):
                warnings_out.append(w)
    else:
        logger.info("[Orchestrator] Output validation passed.")
 
    # ==========================================================================
    # STEP 5 -- Verify chart paths
    # ==========================================================================
    def select_top_high_value_charts(charts, max_charts=6):
        categories = {
            "revenue_trend": ["revenue", "turnover trend"],
            "market_comparison": ["market", "comparison", "hpcl vs", "omc", "bpcl", "iocl", "reliance", "nayara"],
            "inventory_comparison": ["inventory", "closing stock", "opening stock", "stock levels"],
            "turnover_comparison": ["turnover ratio", "turnover comparison", "stock movement"],
            "stock_imbalance": ["imbalance", "stock share", "sales share", "stock vs sales"],
            "product_contribution": ["contribution", "product-wise", "product sales", "sales by product"]
        }
        
        selected = {}
        remaining = []
        
        for chart in charts:
            title, path, sig = chart
            title_lower = title.lower()
            
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
    verified = [(h, p) for h, p, _sig in ordered_charts if os.path.exists(p)]
    logger.info(f"Verified chart paths: {len(verified)}")
 
    # ==========================================================================
    # STEP 6 -- Write document
    # ==========================================================================
    output_path = _resolve_path(_sanitize(title, output_format))
 
    if output_format == "docx":
        generate_docx(body, output_path, title, chart_images=verified, doc_type=doc_type, user_prompt=prompt)
        return output_path
 
    elif output_format == "pdf":
        generate_pdf(body, output_path, title, chart_images=verified, doc_type=doc_type, user_prompt=prompt)
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
 
 
# ==============================================================================
# Helpers
# ==============================================================================
 
def _flatten_multisheet(data: dict) -> dict:
    sheets         = data.get('sheets', {})
    all_candidates = []
    all_samples    = {}
    all_stats      = {}
    all_columns    = []
    all_entity_stats = []
    all_trends     = {}
    all_domain_map = {}
    all_causal_hints = []
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
        # dataset_role and intent are set by the caller after _flatten_multisheet returns
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
    # Robust fallback title based on clean prompt subject
    clean_words = []
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
    for w in re.findall(r'\b\w+\b', prompt.lower()):
        if w not in strip_words and len(w) > 2:
            clean_words.append(w)
    
    subject = " ".join(clean_words[:5])
    if not subject:
        subject = "HPCL Operations"
        
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

    # Always use LLM to generate a professional title, with clear instructions to avoid prompt fragments
    llm_prompt = (
        f"You are a professional title generator for HPCL corporate documents.\n"
        f"Generate a professional, concise title (5-8 words) based on the user's request and files.\n\n"
        f"User Request: {prompt[:300]}\n"
        f"Document Type: {doc_type or intent_hint}\n"
    )
    if content_context:
        llm_prompt += f"Uploaded file details:\n{content_context}\n\n"

    llm_prompt += (
        "STRICT Rules for Title Generation:\n"
        "- The title must be clean, professional, and directly reflect the subject of the document.\n"
        "- NEVER include prompt instructions, persona instructions, or system prompts in the title (e.g., do NOT include 'Act As', 'Senior Management Consultant', 'Prepare', 'Draft', 'Create', etc.).\n"
        "- Reply with ONLY the final title. Do NOT output quotes, explanations, markdown, or punctuation at the end."
    )

    try:
        raw = query_llama(
            llm_prompt,
            output_type="plan",
            has_files=bool(file_names)
        )
        title = raw.strip().strip('"\'').strip('.')
        # Remove any leading markdown formatting like '# Title:' or similar
        title = re.sub(r'^#+\s*', '', title)
        title = re.sub(r'^(title|document title):\s*', '', title, flags=re.IGNORECASE)
        title = title.strip().strip('"\'').strip('.')
        
        # Ensure correct capitalization
        title = _title_case(title)
        
        # Verify the title doesn't contain prompt fragments
        forbidden_fragments = ["act as", "management consultant", "prepare a", "draft a", "generate a"]
        if any(f in title.lower() for f in forbidden_fragments) or len(title) < 5:
            raise ValueError("LLM title contained forbidden prompt fragments or was too short.")
            
        # Clean any prefix the LLM might have generated anyway
        for key in ["file note", "office memorandum", "office notice", "circular", "purchase note", "office order"]:
            title = re.sub(rf'^{key}\s*:?\s*', '', title, flags=re.IGNORECASE)
        title = title.strip()
        
        # Prepend prefix if official doc type
        if doc_type:
            doc_title_map = {
                "file_note":          "File Note: ",
                "office_memorandum":  "Office Memorandum: ",
                "office_notice":      "Office Notice: ",
                "circular":           "Circular: ",
                "purchase_note":      "Purchase Note: ",
                "office_order":       "Office Order: "
            }
            if doc_type in doc_title_map:
                title = f"{doc_title_map[doc_type]}{title}"
                
        return title

    except Exception as e:
        logger.warning(f"Title LLM call failed or returned bad title ({e}), using fallback.")
        prefix = ""
        if doc_type:
            doc_title_map = {
                "file_note":          "File Note: ",
                "office_memorandum":  "Office Memorandum: ",
                "office_notice":      "Office Notice: ",
                "circular":           "Circular: ",
                "purchase_note":      "Purchase Note: ",
                "office_order":       "Office Order: "
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