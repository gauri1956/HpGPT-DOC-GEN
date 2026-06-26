import os
import re
import uuid
import time
import logging
from collections import Counter
 
from data_agent        import extract_data, detect_content_intent
from analysis_agent    import run_analysis, run_insight, _validate_digest as _validate_digest_facts, _strip_digest_tags
from graph_agent       import generate_charts_from_data, generate_charts_from_markers, dedup_charts
from doc_writer        import generate_docx
from pdf_writer        import generate_pdf
from ppt_writer        import generate_ppt
from data_fusion_agent import DataFusionAgent
from planner_agent     import plan_tasks
 
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
 def _calculate_validation_length(text: str) -> int:
    """
    Calculate the length of the text for validation, excluding:
    - Tables (lines containing '|')
    - Embedded charts (e.g. [CHART: ...])
    - Callout block delimiters ('>')
    """
    if not text:
        return 0
    cleaned_lines = []
    for line in text.splitlines():
        s = line.strip()
        if '|' in s:
            continue
        if s.startswith('>'):
            s = s[1:].strip()
        s = re.sub(r'\[CHART:[^\]]*\]', '', s)
        cleaned_lines.append(s)
    
    cleaned_text = "\n".join(cleaned_lines).strip()
    return len(cleaned_text)


def _clean_heading(h: str) -> str:
    """
    Normalize heading for matching by removing leading hashes, markdown tags (*, _),
    numbering (e.g. "1. ", "1.1 "), and trimming/lowercasing.
    """
    if not h:
        return ""
    h_clean = h.lower()
    # Strip markdown headers (e.g., #, ##, ###)
    h_clean = re.sub(r'^#+\s*', '', h_clean)
    # Strip leading numbering/letters patterns like "1. ", "A. ", "1.1. "
    h_clean = re.sub(r'^[a-z0-9\.\-\)]+\s+', '', h_clean)
    # Strip markdown formatting
    h_clean = h_clean.replace('*', '').replace('_', '')
    return h_clean.strip()


def _get_section_contents(body: str) -> dict[str, str]:
    """
    Parse a markdown body and extract content for each heading.
    Maps a normalized heading (e.g., 'executive summary') to its text content.
    Level 3 headings and below are kept within their parent section contents.
    """
    sections = {}
    lines = body.splitlines()
    cur_heading = None
    cur_content = []
    
    for line in lines:
        s = line.strip()
        # Regex to match markdown headings (e.g. ## **Executive Summary** or **## Executive Summary**)
        # Capture 1 to 2 hashes at the start (excluding level 3 headings)
        m = re.match(r'^\s*[\*_]*\s*(#{1,2})\s*[\*_]*(.*)$', s)
        if m:
            heading_text = m.group(2).strip()
            # Clean heading using our global normalization helper
            cleaned_key = _clean_heading(heading_text)
            if cleaned_key:
                if cur_heading:
                    sections[cur_heading] = "\n".join(cur_content).strip()
                cur_heading = cleaned_key
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
    Validate generated body text and return list of warning and info strings.
    Prepend messages with ERROR:, WARNING:, or INFO:.
    """
    warnings = []
    
    # 1. Empty or too short report
    if not body or len(body.strip()) < 200:
        warnings.append("ERROR: Generated body is too short (< 200 chars). Provide more detailed instructions.")
        
    has_headings_error = False
    if doc_type not in _OFFICIAL_DOC_TYPES and '##' not in body:
        warnings.append("ERROR: No section headings (##) found in output. Structure the document with appropriate headers.")
        has_headings_error = True

    sections_dict = _get_section_contents(body)
    has_missing_or_short_section = False

    # 2. Mandatory section length and presence checks
    if doc_type and doc_type in _MIN_SECTION_LENGTHS:
        thresholds = _MIN_SECTION_LENGTHS[doc_type]
        for threshold_heading, min_len in thresholds.items():
            norm_threshold = _clean_heading(threshold_heading)
            found_section_text = None
            
            for h_name, content_text in sections_dict.items():
                if norm_threshold in _clean_heading(h_name):
                    found_section_text = content_text
                    break
            
            if found_section_text is not None:
                actual_len = _calculate_validation_length(found_section_text)
                if actual_len < min_len:
                    warnings.append(
                        f"ERROR: Section '{threshold_heading}' content is too short ({actual_len} chars < {min_len} minimum). Provide more detailed analysis."
                    )
                    has_missing_or_short_section = True
            else:
                warnings.append(
                    f"ERROR: Required section '{threshold_heading}' is missing or has no content (missing mandatory section)."
                )
                has_missing_or_short_section = True

    # 3. Official mandatory checks
    if doc_type in _OFFICIAL_DOC_TYPES:
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
            is_present = False
            if term.startswith("##"):
                norm_term = _clean_heading(term)
                if any(norm_term in _clean_heading(h) for h in sections_dict.keys()):
                    is_present = True
            
            if not is_present and term in body.lower():
                is_present = True
                
            if not is_present:
                warnings.append(f"ERROR: {msg} (missing mandatory section)")
                has_missing_or_short_section = True

    # Generic title check (WARNING)
    generic_titles = ["hpcl document", "untitled document", "report", "learning module",
                      "employee training program", "development program", "kpi dashboard",
                      "training presentation"]
    if title and any(gt == title.lower().strip() for gt in generic_titles):
        warnings.append(f"WARNING: Document title '{title}' appears generic.")

    # Placeholder text check (WARNING)
    if doc_type not in _OFFICIAL_DOC_TYPES:
        placeholders = ['[insert', '[tbd]', '[todo]', '[placeholder]', 'lorem ipsum']
        for ph in placeholders:
            if ph in body.lower():
                warnings.append(f"WARNING: Placeholder text '{ph}' found.")

    # File reference check / Evidence traceability (WARNING)
    all_files_referenced = True
    if doc_type not in _OFFICIAL_DOC_TYPES and file_data:
        for fname, data in file_data.items():
            dname = _display_name(fname).lower()
            cols = []
            if data.get('type') == 'multi_sheet':
                for s in data.get('sheets', {}).values():
                    cols.extend(s.get('columns', []))
            else:
                cols = data.get('columns', [])
            top_cols = [c.lower() for c in cols[:10]]
            
            # Extract significant words from filename (minimum 3 chars, skip generic terms)
            fname_words = [
                w for w in re.findall(r'\b\w{3,}\b', dname)
                if w not in {'data', 'file', 'report', 'sheet', 'xlsx', 'csv', 'txt', 'pdf', 'docx', 'comparison'}
            ]
            
            # Extract unique entity sample values to check for semantic containment
            entity_vals = []
            if data and 'entities' in data:
                for ent_name, ent_info in data['entities'].items():
                    vals = ent_info.get('sample_values', [])
                    for val in vals:
                        val_str = str(val).strip().lower()
                        if len(val_str) > 2 and val_str not in {'nan', 'null', 'none'}:
                            entity_vals.append(val_str)
                            
            found = (
                dname in body.lower() or
                (len(fname_words) > 0 and any(w in body.lower() for w in fname_words)) or
                any(c in body.lower() for c in top_cols if len(c) > 3) or
                any(val in body.lower() for val in entity_vals[:15])
            )
            if not found:
                warnings.append(f"WARNING: File '{fname}' is not referenced in the output (missing evidence).")
                all_files_referenced = False

    # Cross-file comparison / reasoning (WARNING)
    has_cross_file_insight_match = False
    if file_data and len(file_data) > 1 and doc_type not in _OFFICIAL_DOC_TYPES:
        connecting_words = [
            "compare", "correlation", "contradict", "aligned", "alongside",
            "compounded", "across", "versus", "vs", "difference", "variance",
            "higher", "lower", "relative", "comparison", "relationship",
            "cross-file", "overlap", "combined", "synthetic", "correlated"
        ]
        if not any(w in body.lower() for w in connecting_words):
            warnings.append("WARNING: No cross-file comparison terms found (cross-file synthesis missing).")
        
        if cross_file_insights:
            referenced_insights = 0
            for ins in cross_file_insights:
                # 1. Check ID-based match (e.g. INS-001)
                ins_id = getattr(ins, 'id', None)
                if ins_id and ins_id.lower() in body.lower():
                    referenced_insights += 1
                    continue
                
                # 2. Or fallback to matching key words from title and detail
                title_text = getattr(ins, 'title', '')
                detail_text = getattr(ins, 'detail', '')
                words = re.findall(r'\b\w{4,}\b', title_text.lower() + " " + detail_text.lower())
                stop_words = {"file", "dataset", "report", "connection", "numeric", "variance",
                              "insight", "priority", "critical", "high", "medium", "low",
                              "value", "values", "common", "shared", "trends", "trend"}
                keywords = [w for w in words if w not in stop_words]
                if keywords:
                    hits = sum(1 for kw in keywords if kw in body.lower())
                    if hits >= min(1, len(keywords)):
                        referenced_insights += 1
                        
            if referenced_insights == 0:
                warnings.append("WARNING: None of the prioritized cross-file insights were detected in the generated body (confidence not referenced).")
            else:
                has_cross_file_insight_match = True

    # Recommendations checks (WARNING)
    recs_match = re.search(r'##\s*(?:Strategic\s+)?Recommendations\b.*', body, re.IGNORECASE | re.DOTALL)
    if recs_match:
        recs_text = recs_match.group()
        if not any(char.isdigit() for char in recs_text):
            warnings.append("WARNING: Recommendations section lacks numeric data or specific metrics (weak recommendation).")

    # Table header checks for business report (WARNING)
    if doc_type == "business_report" and recs_match:
        recs_text = recs_match.group()
        recs_text_lower = recs_text.lower()
        has_rec = any(k in recs_text_lower for k in ["recommendation", "action", "measure"])
        has_rat = any(k in recs_text_lower for k in ["rationale", "justification", "why"])
        has_out = any(k in recs_text_lower for k in ["outcome", "metric", "kpi", "result"])
        
        missing_structure = []
        if not has_rec:
            missing_structure.append("Recommendation")
        if not has_rat:
            missing_structure.append("Operational Rationale")
        if not has_out:
            missing_structure.append("Expected Outcome & Metrics")
            
        if missing_structure:
            warnings.append(
                f"WARNING: Strategic recommendations missing structured sections: {missing_structure}."
            )

    # Official document specifics
    if doc_type in _OFFICIAL_DOC_TYPES:
        if "subject:" not in body.lower() and not any("subject:" in s.lower() for s in body.splitlines()[:15]):
            warnings.append("ERROR: Official document missing required 'Subject:' header.")
            has_missing_or_short_section = True
 
        sig_indicators = [
            "prepared by", "sd/-", "approved by", "recommended by", "submitted for approval",
            "by order", "by order of", "signature", "designation", "department", "authority", "signatory"
        ]
        if not any(ind in body.lower() for ind in sig_indicators):
            warnings.append("ERROR: No approval authority or signatory block detected (missing signatory).")
            has_missing_or_short_section = True
 
        if doc_type in ["file_note", "office_memorandum", "circular"]:
            bullet_count = body.count("\n- ") + body.count("\n* ")
            if bullet_count > 5:
                warnings.append(
                    f"WARNING: Official doc '{doc_type}' should use numbered paragraphs, not bullet points."
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
                    f"WARNING: Purchase Note contains template instruction text: {found_leftovers}."
                )

    # Invented names check (WARNING)
    honorific_match = re.search(
        r'\b(shri|smt|mr|ms|dr)\.?\s+[a-zA-Z]{2,}\s+[a-zA-Z]{2,}',
        body, re.IGNORECASE
    )
    if honorific_match:
        warnings.append(
            f"WARNING: Potential invented person name: '{honorific_match.group()}'."
        )

    # Dynamic date check (WARNING)
    import datetime
    today = datetime.date.today()
    allowed_dates = set()
    for fmt in [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y",
        "%d.%m.%Y", "%Y.%m.%d", "%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"
    ]:
        s_date = today.strftime(fmt).lower()
        allowed_dates.add(s_date)
        parts = re.split(r'([/\-\.\s,]+)', s_date)
        clean_parts = []
        for p in parts:
            if p.isdigit():
                clean_parts.append(str(int(p)))
            else:
                clean_parts.append(p)
        allowed_dates.add("".join(clean_parts))

    date_patterns = re.findall(r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{4}\b', body)
    for dt in date_patterns:
        dt_clean = dt.strip().lower()
        if dt_clean in allowed_dates:
            continue
        is_today = False
        for sep in ['/', '-', '.']:
            if sep in dt_clean:
                parts = dt_clean.split(sep)
                if len(parts) == 3:
                    try:
                        p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])
                        if (p0 == today.day and p1 == today.month and p2 == today.year) or \
                           (p0 == today.month and p1 == today.day and p2 == today.year) or \
                           (p0 == today.year and p1 == today.month and p2 == today.day) or \
                           (p0 == today.year and p1 == today.day and p2 == today.month):
                            is_today = True
                            break
                    except ValueError:
                        pass
        if is_today:
            continue
            
        if dt not in prompt:
            in_files = False
            if file_data:
                in_files = any(dt in str(fdata) for fdata in file_data.values())
            if not in_files:
                warnings.append(f"WARNING: Potential invented date: '{dt}'.")

    # Trend contradictions (WARNING)
    if doc_type not in _OFFICIAL_DOC_TYPES and file_data:
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
                                    f"WARNING: Trend contradiction for '{col}' in '{fname}'. "
                                    f"Data=UP but body has negative language."
                                )
                        elif direction == "down":
                            positives = ["increase", "grow", "rose", "upward", "climb", "expansion"]
                            if any(pos in window for pos in positives) and not any(
                                neg in window for neg in ["decrease", "decline", "drop", "fell", "downward", "shrank", "reduced"]
                            ):
                                warnings.append(
                                    f"WARNING: Trend contradiction for '{col}' in '{fname}'. "
                                    f"Data=DOWN but body has positive language."
                                )
                        idx += len(clean_col)

    # Invented metrics validation (WARNING)
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
                f"WARNING: Potential invented metrics in no-file generation: {', '.join(found_metrics)}."
            )

    # Positive INFO milestones (Non-flooding, meaningful milestones only)
    if not has_headings_error and not has_missing_or_short_section:
        warnings.append("INFO: All mandatory sections present.")
    if has_cross_file_insight_match:
        warnings.append("INFO: Cross-file reasoning verified.")
    if file_data and all_files_referenced:
        warnings.append("INFO: Evidence traceability verified.")
        
    # Append final milestone if completely clean of any errors or warnings
    has_errors_or_warnings = any(w.startswith("ERROR:") or w.startswith("WARNING:") for w in warnings)
    if not has_errors_or_warnings:
        warnings.append("INFO: Validator completed successfully.")

    return warnings
 
# ==============================================================================
# DIGEST VALIDATION
# Lightweight inline check: does each digest bullet have a data reference?
# NOTE: This is an alias/wrapper. The core logic lives in analysis_agent.py
# as _validate_digest_facts (imported above) so it can also be used
# independently. The orchestrator calls _validate_digest() here.
# ==============================================================================
 
def _validate_digest(digest_text: str, file_name: str) -> list[str]:
    """
    Check each bullet in a digest for unsupported numeric claims.
    Delegates to analysis_agent._validate_digest_facts which checks for
    [src:] tags and known domain keywords as fallback markers.
 
    Called on the RAW tagged digest (before stripping) so [src:] presence
    is visible. Returns list of warning strings (empty = all good).
    """
    return _validate_digest_facts(digest_text, file_name)
 
 
# ==============================================================================
# SELF-CORRECTION
# One retry pass when severe validation issues are detected.
# After correction, re-extracts charts from the corrected body for non-official
# doc types so ordered_charts stays consistent.
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
    cross_file_insights: list = None,
    file_data: dict = None,
    entity_map: dict = None,
    links: list = None,
    ranked_files: list = None,
) -> tuple[str, list]:
    """
    Attempt one self-correction LLM call when severe structural warnings
    are found in the generated body (missing mandatory sections, template
    leftovers in purchase notes, or zero cross-file connections).
 
    Only fires for severe warnings — not for minor style notes.
 
    Returns:
        (corrected_body, new_charts)
        corrected_body: corrected text (or original if correction failed/unneeded)
        new_charts:     charts extracted from corrected body for non-official docs
                        (empty list for official docs or if no correction occurred)
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
        return body, []   # nothing severe — skip retry
 
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
            cross_file_insights=cross_file_insights,
            file_data=file_data,
            entity_map=entity_map,
            links=links,
            ranked_files=ranked_files,
        )
        logger.info("[Orchestrator] Self-correction LLM call completed.")
 
        # Re-extract charts from corrected body for non-official docs.
        # This keeps ordered_charts consistent with the corrected text.
        new_charts = []
        if doc_type not in _OFFICIAL_DOC_TYPES:
            corrected_body, new_charts = generate_charts_from_markers(corrected_raw)
            logger.info(
                f"[Orchestrator] Self-correction: extracted {len(new_charts)} "
                f"chart marker(s) from corrected body."
            )
        else:
            corrected_body = corrected_raw
 
        return corrected_body, new_charts
 
    except Exception as e:
        logger.warning(f"[Orchestrator] Self-correction failed: {e}. Using original body.")
        return body, []
 
 
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
    # STEP 1c -- Use planner to lock doc_type when not set from prompt
    # ==========================================================================
    if doc_type is None and file_data:
        intent_list = list(file_intents.values())
        plan = plan_tasks(
            prompt       = prompt,
            file_names   = list(file_data.keys()),
            file_intents = intent_list,
        )
        planner_doc_type = plan.get("doc_type")
        if planner_doc_type and planner_doc_type != "informational":
            doc_type = planner_doc_type
            logger.info(f"[Orchestrator] doc_type locked by planner: {doc_type}")
        else:
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
        doc_type = None
 
    # ==========================================================================
    # STEP 1d -- DataFusionAgent
    # ==========================================================================
    cross_file_context  = ""
    ranked_file_names: list[str] = list(file_data.keys())
    cross_file_insights = []
    entity_map = None
    links = None
    ranked_files = None
 
    if has_files and file_data:
        logger.info("[Orchestrator] Running DataFusionAgent ...")
        try:
            fusion_agent  = DataFusionAgent(file_data, user_prompt=prompt)
            fusion_result = fusion_agent.run()
            cross_file_insights = fusion_result.insights
            entity_map = fusion_result.entity_map
            links = fusion_result.links
            ranked_files = fusion_result.ranked_files
 
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
                # Cached digests are already stripped (tags removed before caching below)
                clean_digest = cached_entry["digest_text"]
                detected_intent = cached_entry["detected_intent"]
                cached_charts = cached_entry.get("charts", [])
                
                digest_results.append((clean_digest, detected_intent))
                
                if doc_type not in _OFFICIAL_DOC_TYPES:
                    processed_charts = []
                    for item in cached_charts:
                        if isinstance(item, dict):
                            processed_charts.append(item)
                        else:
                            title_c = item[0]
                            path_c = item[1]
                            sig_c = item[2]
                            col_c = item[3] if len(item) > 3 else ""
                            processed_charts.append({
                                "title": title_c,
                                "path": path_c,
                                "signature": sig_c,
                                "column": col_c
                            })
                    
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
                tagged_digest_text, detected_intent = run_analysis(
                    flat,
                    instruction=f"{prompt} — analysing {_display_name(name)}",
                    intent=intent,
                    dataset_role=dataset_role,
                    causal_hints=causal_hints,
                    prior_digests_summary=prior_digests_summary
                )
            else:
                tagged_digest_text, detected_intent = run_analysis(
                    data,
                    instruction=f"{prompt} — analysing {_display_name(name)}",
                    intent=intent,
                    dataset_role=dataset_role,
                    causal_hints=causal_hints,
                    prior_digests_summary=prior_digests_summary
                )
 
            # FIX: Validate on the RAW tagged digest (tags visible for [src:] checks),
            # then strip tags before caching and passing to run_insight().
            # This is the correct separation: validate -> strip -> cache/use.
            digest_warnings = _validate_digest(tagged_digest_text, name)
            if digest_warnings:
                for dw in digest_warnings:
                    logger.warning(dw)
                    if isinstance(warnings_out, list):
                        warnings_out.append(dw)
 
            # Strip [src:]/[conf:] tags AFTER validation — clean digest for LLM use
            digest_text = _strip_digest_tags(tagged_digest_text)
 
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
                for chart in file_charts:
                    if isinstance(chart, dict):
                        sig = chart.get("signature")
                        serializable_charts.append({
                            "title": chart.get("title", ""),
                            "path": chart.get("path", ""),
                            "signature": list(sig) if sig else None,
                            "column": chart.get("column", "")
                        })
                    else:
                        title_c, path_c, sig_c, col_c = chart[0], chart[1], chart[2], chart[3] if len(chart) > 3 else ""
                        serializable_charts.append({
                            "title": title_c,
                            "path": path_c,
                            "signature": list(sig_c) if sig_c else None,
                            "column": col_c
                        })
                
                # Cache the STRIPPED digest (no [src:]/[conf:] tags)
                cache_data["digests"][name] = {
                    "digest_text": clean_digest,
                    "detected_intent": detected_intent,
                    "charts": serializable_charts
                }
            else:
                # For official docs, no chart extraction needed — cache stripped digest
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
            official_format_spec=official_format_spec,
            cross_file_insights=cross_file_insights,
            file_data=file_data,
            entity_map=entity_map,
            links=links,
            ranked_files=ranked_files,
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
 
    # -- STEP 4d -- Self-correction (has_files path only) ----------------------
    if has_files and validation_warnings and digest_results:
        corrected, correction_charts = _self_correct_body(
            body                 = body,
            warnings             = validation_warnings,
            prompt               = prompt,
            output_format        = output_format,
            doc_type             = doc_type,
            official_format_spec = official_format_spec,
            digest_results       = digest_results,
            cross_file_context   = cross_file_context,
            cross_file_insights  = cross_file_insights,
            file_data            = file_data,
            entity_map           = entity_map,
            links                = links,
            ranked_files         = ranked_files,
        )
        if corrected != body:
            body = corrected
 
            # FIX: For non-official docs, merge any charts from corrected body.
            # Dedup against existing seen_signatures to avoid duplicates.
            if correction_charts and doc_type not in _OFFICIAL_DOC_TYPES:
                deduped_correction, seen_signatures = dedup_charts(
                    correction_charts, seen_signatures
                )
                ordered_charts.extend(deduped_correction)
                logger.info(
                    f"[Orchestrator] Self-correction added {len(deduped_correction)} "
                    f"new chart(s) from corrected body."
                )
 
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
            if isinstance(chart, dict):
                title_c = chart.get("title", "")
            else:
                title_c = chart[0]
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
    verified = []
    for chart in ordered_charts:
        if isinstance(chart, dict):
            p = chart.get("path", "")
            if os.path.exists(p):
                verified.append(chart)
        else:
            p = chart[1]
            if os.path.exists(p):
                verified.append({
                    "title": chart[0],
                    "path": p,
                    "signature": chart[2],
                    "column": chart[3] if len(chart) > 3 else ""
                })
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