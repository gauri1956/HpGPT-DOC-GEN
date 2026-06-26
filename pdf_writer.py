import os
import re
import logging

from parser_utils import parse_sections, extract_table, strip_table_from_text
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, black
from reportlab.lib.units import inch

PAGE_WIDTH, PAGE_HEIGHT = LETTER
LEFT_MARGIN = RIGHT_MARGIN = inch * 0.7
TOP_MARGIN  = BOTTOM_MARGIN = inch * 0.7
LINE_HEIGHT = 17
CHART_H     = 2.4 * inch

TITLE_COLOR  = HexColor("#00205b")
FOOTER_COLOR = HexColor("#e4002b")
BODY_COLOR   = black
TABLE_HDR_BG = HexColor("#00205b")
TABLE_ALT_BG = HexColor("#f2f4f8")

CALLOUT_BG     = HexColor("#eef1f8")
CALLOUT_BORDER = HexColor("#00205b")
WARNING_BG     = HexColor("#fff4f4")
WARNING_BORDER = HexColor("#e4002b")


def _apply_theme(org_name):
    global TITLE_COLOR, FOOTER_COLOR, TABLE_HDR_BG, CALLOUT_BORDER, WARNING_BORDER
    ol = str(org_name).lower()
    if "bharat petroleum" in ol or "bpcl" in ol:
        TITLE_COLOR    = HexColor("#0072ce")
        FOOTER_COLOR   = HexColor("#ffc72c")
        TABLE_HDR_BG   = HexColor("#0072ce")
        CALLOUT_BORDER = HexColor("#0072ce")
    elif "indian oil" in ol or "iocl" in ol or "indianoil" in ol:
        TITLE_COLOR    = HexColor("#ff6600")
        FOOTER_COLOR   = HexColor("#0033aa")
        TABLE_HDR_BG   = HexColor("#ff6600")
        CALLOUT_BORDER = HexColor("#ff6600")
    else:
        TITLE_COLOR    = HexColor("#00205b")
        FOOTER_COLOR   = HexColor("#e4002b")
        TABLE_HDR_BG   = HexColor("#00205b")
        CALLOUT_BORDER = HexColor("#00205b")


USABLE_W = PAGE_WIDTH  - LEFT_MARGIN - RIGHT_MARGIN
USABLE_H = PAGE_HEIGHT - TOP_MARGIN  - BOTTOM_MARGIN

_CHART_LINE = re.compile(r'^\s*\[CHART:', re.IGNORECASE)

_SECTION_TYPE_MAP = [
    (re.compile(r'\bexecutive\s+summary\b',          re.IGNORECASE), "executive_summary"),
    (re.compile(r'\brisk\b|\bwarning\b|\bcaution\b', re.IGNORECASE), "risk_section"),
    (re.compile(r'\bkpi\b|\bdashboard\b|\bscorecard\b', re.IGNORECASE), "kpi_section"),
    (re.compile(r'\brecommendations?\b',             re.IGNORECASE), "recommendations"),
    (re.compile(r'\blearning\s+objectives?\b',       re.IGNORECASE), "objectives"),
    (re.compile(r'\bcase\s+study\b',                 re.IGNORECASE), "case_study"),
    (re.compile(r'\bassessment\b|\bquiz\b',          re.IGNORECASE), "assessment"),
    (re.compile(r'\bsafety\b|\bwarning\b|\bstop\b',  re.IGNORECASE), "safety"),
    (re.compile(r'\bchecklist\b',                    re.IGNORECASE), "checklist"),
]


def _detect_section_type(heading: str) -> str:
    for pattern, stype in _SECTION_TYPE_MAP:
        if pattern.search(heading):
            return stype
    return "body"


_OFFICIAL_DOC_TYPES = {
    "file_note", "office_memorandum", "office_notice",
    "circular", "purchase_note", "office_order"
}


# ── PATCHED: parse_official_header ────────────────────────────────────────────
# Changes vs original:
#   1. Added `doc_type` parameter
#   2. Footer scanner now only activates AFTER the last ## section + 20-line
#      buffer — prevents stripping approval lines inside Section 7 of a
#      Purchase Note (or any section body that contains colon-separated lines)
#   3. `purchase_note` skips the footer scan entirely (approval chain is inline)
#   4. Added 'note_no', 'notice_no', 'order_no', 'circular_no' to header key
#      normalisation set (was missing previously)

def parse_official_header(body_text, doc_type=None, user_prompt=None):
    """
    Extracts header fields and returns (header_dict, remaining_body_text).

    FIX: The footer scanner (Prepared by / Approved by etc.) now only
    activates AFTER the last ## section has ended + a 20-line buffer.
    This prevents stripping approval lines that appear INSIDE Section 7
    of a Purchase Note (or any deeply-nested section body).
    """
    header_fields = {}
    lines         = body_text.splitlines()
    body_lines    = []
    in_header     = True
    non_empty_count = 0

    for line in lines:
        sline = line.strip()
        if in_header:
            if sline:
                non_empty_count += 1

            if (sline.startswith('##')
                    or re.match(r'^[=\-_*]{3,}$', sline)
                    or non_empty_count > 15):
                in_header = False
                if sline.startswith('##'):
                    body_lines.append(line)
                continue

            if sline.startswith('#'):
                header_fields['title'] = sline.lstrip('#').strip()
            elif ':' in sline:
                parts       = sline.split(':', 1)
                k_candidate = parts[0].strip()
                v_candidate = parts[1].strip()
                k_norm = (k_candidate.lower()
                                      .replace(' ', '_')
                                      .replace('.', ''))
                if k_norm in {
                    'ref', 'ref_no', 'file_no', 'no',
                    'note_no', 'notice_no', 'order_no', 'circular_no',
                    'date', 'subject', 'sub', 'to', 'from', 'through',
                    'department', 'dept', 'title',
                    'reference', 'reference_no',
                    'organization', 'company', 'org',
                }:
                    header_fields[k_norm] = v_candidate
                else:
                    in_header = False
                    body_lines.append(line)
            elif sline != '':
                in_header = False
                body_lines.append(line)
        else:
            body_lines.append(line)

    if not body_lines:
        body_lines = lines

    full_text = "\n".join(lines[:15])

    # Extract ref / file number
    for pattern in [
        r'(?:File\s+No\.|Note\s+No\.|Notice\s+No\.|Circular\s+No\.|Order\s+No\.|No\.)\s*:\s*([^\n\t]+)',
        r'(?:Circular\s+No\.|Order\s+No\.)\s*([^\n\t]+)',
    ]:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if 'date' in val.lower():
                val = re.split(r'\bdate\b', val,
                               flags=re.IGNORECASE)[0].strip().rstrip(':').strip()
            header_fields['ref_no'] = val
            break

    m_date = re.search(r'Date\s*:\s*([^\n\t\r]+)', full_text, re.IGNORECASE)
    if m_date:
        header_fields['date'] = m_date.group(1).strip()

    m_subj = re.search(r'Subject\s*:\s*([^\n\t\r]+)', full_text, re.IGNORECASE)
    if m_subj:
        header_fields['subject'] = m_subj.group(1).strip()

    m_dept = re.search(r'Department\s*:\s*([^\n\t\r]+)', full_text, re.IGNORECASE)
    if m_dept:
        header_fields['department'] = m_dept.group(1).strip()

    m_to = re.search(r'\bTo\s*:\s*([^\n\t\r]+)', full_text, re.IGNORECASE)
    if m_to:
        header_fields['to'] = m_to.group(1).strip()

    m_from = re.search(r'\bFrom\s*:\s*([^\n\t\r]+)', full_text, re.IGNORECASE)
    if m_from:
        header_fields['from'] = m_from.group(1).strip()

    # Clean bracketed placeholders like "[file number]" or "[date]"
    for k in ['ref_no', 'date', 'subject', 'department', 'to', 'from']:
        if k in header_fields and (
                '[' in header_fields[k] or ']' in header_fields[k]):
            header_fields[k] = ""

    # Scrub invented dates
    if 'date' in header_fields and header_fields['date']:
        date_val = header_fields['date']
        if '_' not in date_val and user_prompt:
            prompt_lower = user_prompt.lower()
            words = re.findall(r'\b\w+\b', date_val.lower())
            found = False
            for w in words:
                if w in prompt_lower:
                    found = True
                    break
            if not found:
                header_fields['date'] = "__________________"

    # ── Footer scanner (signature / distribution lines) ─────────────────────
    # PATCH: Only scan lines that appear AFTER the last ## section heading
    # plus a 20-line safety buffer.  This stops the scanner from eating
    # colon-separated lines that legitimately belong inside a section body
    # (e.g. the Approval Chain table inside Purchase Note § 7).
    # For purchase_note, skip the footer scan entirely.

    footer_keys = {
        'prepared_by', 'recommended_by', 'approved_by',
        'submitted_for_approval_of', 'submitted_for_approval_to',
        'submitted_for_approval', 'submitted_by', 'signatory',
        'by_order_of', 'copy_to', 'distribution',
    }

    last_section_idx = -1
    for i, line in enumerate(body_lines):
        if line.strip().startswith('##'):
            last_section_idx = i

    skip_footer_scan = (doc_type == "purchase_note")

    final_body_lines = []
    if skip_footer_scan or last_section_idx == -1:
        final_body_lines = body_lines
    else:
        footer_scan_start = last_section_idx + 20

        for i, line in enumerate(body_lines):
            sline = line.strip()
            if (i >= footer_scan_start
                    and ':' in sline
                    and not sline.startswith('##')
                    and not sline.startswith('#')):
                parts       = sline.split(':', 1)
                k_candidate = parts[0].strip()
                v_candidate = parts[1].strip()
                k_norm = (k_candidate.lower()
                                      .replace(' ', '_')
                                      .replace('.', ''))

                if k_norm in footer_keys:
                    if (k_norm == 'submitted_for_approval_of'
                            and 'approved_by' not in header_fields):
                        header_fields['approved_by'] = v_candidate
                    elif (k_norm == 'submitted_by'
                          and 'prepared_by' not in header_fields):
                        header_fields['prepared_by'] = v_candidate
                    else:
                        header_fields[k_norm] = v_candidate
                    continue
            final_body_lines.append(line)

    if not final_body_lines:
        final_body_lines = body_lines

    return header_fields, "\n".join(final_body_lines).strip()


# ── Official PDF header block ──────────────────────────────────────────────────

def _draw_official_pdf_header(c, metadata, doc_type, y):
    org_candidate = (metadata.get("organization")
                     or metadata.get("company")
                     or metadata.get("org"))
    ref_candidate = (metadata.get("ref_no")
                     or metadata.get("file_no")
                     or metadata.get("no"))

    if not org_candidate:
        org_candidate = "__________________________________________________"

    org = org_candidate
    ref = ref_candidate or "__________________"

    # Company name
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(TITLE_COLOR)
    c.drawCentredString(PAGE_WIDTH / 2, y, org.upper())
    y -= 18

    # Department
    dept = (metadata.get("department")
            or metadata.get("department_name")
            or "Corporate Office")
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(HexColor("#333333"))
    c.drawCentredString(PAGE_WIDTH / 2, y, dept)
    y -= 12

    c.setStrokeColor(HexColor("#cccccc"))
    c.setLineWidth(0.8)
    c.line(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y)
    y -= 14

    # Ref & Date
    date = metadata.get("date") or "__________________"
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(BODY_COLOR)
    c.drawString(LEFT_MARGIN, y, f"Ref: {ref}")
    c.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, y, f"Date: {date}")
    y -= 8
    c.line(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y)
    y -= 20

    # Document type title
    doc_title_map = {
        "file_note":         "FILE NOTE",
        "office_memorandum": "OFFICE MEMORANDUM",
        "office_notice":     "OFFICE NOTICE",
        "circular":          "CIRCULAR",
        "purchase_note":     "PURCHASE NOTE",
        "office_order":      "OFFICE ORDER",
    }
    dtype_name = doc_title_map.get(doc_type,
                                   doc_type.upper().replace("_", " "))
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(TITLE_COLOR)
    c.drawCentredString(PAGE_WIDTH / 2, y, dtype_name)
    y -= 20

    # Subject
    subj = metadata.get("subject") or "Official Directives"
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(BODY_COLOR)
    wrapped_subj = _wrap_text_list(c, f"Subject: {subj}",
                                   USABLE_W, "Helvetica-Bold", 11)
    for wl in wrapped_subj:
        c.drawString(LEFT_MARGIN, y, wl)
        c.setStrokeColor(BODY_COLOR)
        c.setLineWidth(0.8)
        c.line(LEFT_MARGIN, y - 2,
               LEFT_MARGIN + c.stringWidth(wl, "Helvetica-Bold", 11), y - 2)
        y -= 16
    y -= 4

    # To / From / Through routing
    for field in ["to", "through", "from"]:
        val = metadata.get(field)
        if val:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(LEFT_MARGIN, y, f"{field.capitalize()}:")
            c.setFont("Helvetica", 10)
            c.drawString(LEFT_MARGIN + 60, y, val)
            y -= 14

    if any(metadata.get(f) for f in ["to", "through", "from"]):
        y -= 6
        c.setStrokeColor(HexColor("#dddddd"))
        c.setLineWidth(0.5)
        c.line(LEFT_MARGIN, y, PAGE_WIDTH - RIGHT_MARGIN, y)
        y -= 14

    return y


# ── Official PDF footer block ──────────────────────────────────────────────────
# FIX: Was truncated in previous version — the copy_to/distribution block
# had no closing code and never returned y, causing a silent AttributeError
# that swallowed the entire signature/distribution section at runtime.

def _draw_official_pdf_footer(c, metadata, y):
    """
    Draw the official document footer: signature block + distribution list.
    Returns the final y position after rendering.
    """
    needed = 120
    if y < BOTTOM_MARGIN + needed:
        draw_footer(c)
        c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN

    y -= 30
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(BODY_COLOR)

    sig_lines = []
    for k, label in [
        ("prepared_by",    "Prepared by"),
        ("recommended_by", "Recommended by"),
        ("approved_by",    "Approved by"),
    ]:
        v = metadata.get(k)
        if v:
            sig_lines.append(f"({label})")
            sig_lines.append(v)
            sig_lines.append("")

    if not sig_lines:
        v_sig = metadata.get("signatory") or metadata.get("by_order_of")
        if v_sig:
            sig_lines.extend(["", f"{v_sig}"])

    for s_line in sig_lines:
        if y < BOTTOM_MARGIN + 30:
            draw_footer(c)
            c.showPage()
            y = PAGE_HEIGHT - TOP_MARGIN
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(BODY_COLOR)
        if s_line:
            c.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, y, s_line)
        y -= 12

    # Distribution / Copy To block
    for k, label in [("copy_to", "Copy to:"), ("distribution", "Distribution:")]:
        val = metadata.get(k)
        if val:
            y -= 10
            if y < BOTTOM_MARGIN + 50:
                draw_footer(c)
                c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(BODY_COLOR)
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(HexColor("#555555"))
            c.drawString(LEFT_MARGIN, y, label)
            y -= 12
            c.setFont("Helvetica", 9)
            c.setFillColor(BODY_COLOR)
            # Distribution may be comma-separated — split and render each entry
            entries = [e.strip() for e in val.split(',') if e.strip()]
            for entry in entries:
                if y < BOTTOM_MARGIN + 30:
                    draw_footer(c)
                    c.showPage()
                    y = PAGE_HEIGHT - TOP_MARGIN
                    c.setFont("Helvetica", 9)
                    c.setFillColor(BODY_COLOR)
                c.drawString(LEFT_MARGIN + 12, y, f"• {entry}")
                y -= 12

    return y


# ── Main entry point ───────────────────────────────────────────────────────────

def generate_pdf(body, output_path="outputs/output.pdf",
                 title="AI Generated PDF", chart_images=None, doc_type=None, user_prompt=None):
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True,
    )
    chart_images = _dedupe_charts(list(chart_images or []))
    display_title = _clean_title(title)
    body = _strip_chart_placeholders(body)
    body = _clean_bracketed_placeholders(body)

    is_official = doc_type in _OFFICIAL_DOC_TYPES
    metadata    = {}

    c = canvas.Canvas(output_path, pagesize=LETTER)

    org_name = ""
    if is_official:
        # Pass doc_type so purchase_note skips footer scan
        metadata, body = parse_official_header(body, doc_type=doc_type, user_prompt=user_prompt)
        org_name = (metadata.get('organization')
                    or metadata.get('company')
                    or metadata.get('org') or "")
        if not org_name:
            bl = body.lower()
            if "bpcl" in bl or "bharat petroleum" in bl:
                org_name = "BHARAT PETROLEUM CORPORATION LIMITED"
            elif "iocl" in bl or "indian oil" in bl or "indianoil" in bl:
                org_name = "INDIAN OIL CORPORATION LIMITED"
            elif "ongc" in bl or "oil and natural gas" in bl:
                org_name = "OIL AND NATURAL GAS CORPORATION LIMITED"
            elif "gail" in bl:
                org_name = "GAIL (INDIA) LIMITED"
            elif "hpcl" in bl or "hindustan petroleum" in bl:
                org_name = "HINDUSTAN PETROLEUM CORPORATION LIMITED"
        metadata['organization'] = org_name
    else:
        _draw_cover(c, display_title)
        c.showPage()
        org_name = display_title

    _apply_theme(org_name)

    y = PAGE_HEIGHT - TOP_MARGIN

    if is_official:
        y = _draw_official_pdf_header(c, metadata, doc_type, y)

    sections = parse_sections(clean_text(body))
    assignments, unplaced_idx = _assign_charts_to_sections(sections, chart_images)

    for sec_idx, sec in enumerate(sections):
        level, heading, content, table = \
            sec['level'], sec['heading'], sec['body'], sec['table']

        if y < BOTTOM_MARGIN + 120:
            draw_footer(c)
            c.showPage()
            y = PAGE_HEIGHT - TOP_MARGIN

        sec_type = _detect_section_type(heading) if heading else "body"

        # Heading
        if level == 1:
            y = _draw_heading(c, heading, y, 15, TITLE_COLOR, underline=True)
        elif level == 2:
            y = _draw_section_heading(c, heading, y, sec_type)
        elif level == 3:
            y = _draw_heading(c, heading, y, 11, HexColor("#333333"))

        # Strip markdown table if present to prevent rendering raw markdown tables inside callout/list blocks
        tbl = extract_table(content)
        clean_content = strip_table_from_text(content) if tbl else content

        # Content — section-type-aware
        if sec_type == "executive_summary":
            y = _draw_callout_block(c, clean_content, y,
                                    bg=CALLOUT_BG, border=CALLOUT_BORDER,
                                    label="EXECUTIVE SUMMARY")
        elif sec_type in ("risk_section", "safety"):
            y = _draw_callout_block(c, clean_content, y,
                                    bg=WARNING_BG, border=WARNING_BORDER,
                                    label="⚠ ATTENTION")
        elif sec_type == "kpi_section" and tbl:
            y = draw_kpi_table(c, tbl, y)
        elif sec_type == "checklist":
            y = _draw_checklist(c, clean_content, y)
        elif sec_type == "objectives":
            y = _draw_numbered_list(c, clean_content, y, accent=TITLE_COLOR)
        elif sec_type == "recommendations":
            y = _draw_numbered_list(c, clean_content, y, accent=FOOTER_COLOR)
        elif sec_type == "assessment":
            y = _draw_assessment(c, clean_content, y)
        else:
            y = draw_body(c, clean_content, y)

        # Draw the table if one was extracted and we didn't draw it as a KPI table
        if tbl and sec_type != "kpi_section":
            y -= 8
            y = draw_table(c, tbl, y)

        # Inline charts
        for idx in assignments.get(sec_idx, []):
            chart = chart_images[idx]
            img_path = chart.get("path", "") if isinstance(chart, dict) else chart[1]
            y = _place_chart(c, img_path, y)

        y -= 8

    # Appendix: unplaced charts
    if unplaced_idx:
        draw_footer(c)
        c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN
        y = _draw_heading(c, "Charts & Visualisations", y, 15,
                          TITLE_COLOR, underline=True)
        y -= 8
        for idx in unplaced_idx:
            chart = chart_images[idx]
            if isinstance(chart, dict):
                ct = chart.get("title", "")
                img_path = chart.get("path", "")
            else:
                ct = chart[0]
                img_path = chart[1]
            label  = _clean_col_name(ct)
            img_h  = _img_h(img_path)
            needed = img_h + 26
            if y - needed < BOTTOM_MARGIN + 50:
                draw_footer(c)
                c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(TITLE_COLOR)
            c.drawString(LEFT_MARGIN, y, label)
            y -= 14
            y = draw_image(c, img_path, y)
            y -= 14

    if is_official:
        y = _draw_official_pdf_footer(c, metadata, y)

    draw_footer(c)
    if c._pageNumber == 0:
        c.showPage()
    c.save()
    logging.info(f"PDF saved to: {output_path}")
    print(f"PDF saved to: {output_path}")


# ── Title cleaning ─────────────────────────────────────────────────────────────

def _clean_title(title):
    t = re.sub(r'^[#*\-_=\s"\']+|[#*\-_=\s"\']+$', '', title).strip()
    skip = {'a', 'an', 'the', 'of', 'in', 'on', 'with', 'and', 'or', 'for',
            'to', 'by', 'give', 'me', 'make', 'create', 'generate', 'write',
            'build', 'produce', 'get'}
    words = t.split()
    while words and words[0].lower() in skip:
        words.pop(0)
    result = []
    for i, w in enumerate(words):
        if i == 0 or i == len(words) - 1 or w.lower() not in skip:
            result.append(w.capitalize())
        else:
            result.append(w.lower())
    return ' '.join(result) if result else title.strip()


# ── Chart placeholder stripping ───────────────────────────────────────────────

def _strip_chart_placeholders(text):
    return '\n'.join(
        line for line in text.splitlines()
        if not _CHART_LINE.match(line)
    )


def _clean_bracketed_placeholders(text):
    def replacer(match):
        content = match.group(1).lower().strip()
        if content.startswith('chart:') or (len(content) <= 4 and content.isupper()):
            return match.group(0)
        placeholder_words = {
            'tbd', 'insert', 'todo', 'placeholder', 'draft', 'unknown', 'xxxx', 'yyyy',
            'subject', 'ref', 'date', 'name', 'cost', 'amount', 'budget', 'value', 'price',
            'officer', 'authority', 'sign', 'signature', 'designation', 'approved', 'recommended',
            'item', 'service', 'description', 'spec', 'quantity', 'qty'
        }
        if any(w in content for w in placeholder_words) or len(content) > 4:
            if 'cost' in content or 'amount' in content or 'budget' in content or 'price' in content:
                return "Rs. __________________"
            if 'date' in content:
                return "__________________"
            return "__________________"
        return match.group(0)
    return re.sub(r'\[([^\]]+)\]', replacer, text)


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedupe_charts(chart_images):
    seen_paths, seen_titles, result = set(), set(), []
    for chart in chart_images:
        if isinstance(chart, dict):
            ct = chart.get("title", "")
            path = chart.get("path", "")
            col = chart.get("column", "")
        else:
            ct = chart[0]
            path = chart[1]
            col = chart[2] if len(chart) > 2 else ""
            
        norm_title = ct.lower().strip()
        if path in seen_paths or norm_title in seen_titles:
            continue
        seen_paths.add(path)
        seen_titles.add(norm_title)
        
        if isinstance(chart, dict):
            result.append(chart)
        else:
            result.append({
                "title": ct,
                "path": path,
                "column": col
            })
    return result


# ── Cover page ────────────────────────────────────────────────────────────────

def _draw_cover(c, title):
    c.setFillColor(TITLE_COLOR)
    c.rect(0, PAGE_HEIGHT - inch * 1.8, PAGE_WIDTH, inch * 1.8, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(HexColor("#ffffff"))
    c.drawString(LEFT_MARGIN, PAGE_HEIGHT - inch * 0.9, "HPGPT")
    c.setFillColor(FOOTER_COLOR)
    c.rect(0, PAGE_HEIGHT - inch * 2.0, PAGE_WIDTH, inch * 0.2, fill=1, stroke=0)

    clean = re.sub(r'^[#*\-_=\s]+|[#*\-_=\s]+$', '', title).strip()
    words, lines, cur = clean.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if c.stringWidth(t, "Helvetica-Bold", 24) <= USABLE_W:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)

    y = (PAGE_HEIGHT / 2) + len(lines) * 18
    for line in lines:
        c.setFont("Helvetica-Bold", 24)
        c.setFillColor(TITLE_COLOR)
        c.drawCentredString(PAGE_WIDTH / 2, y, line)
        y -= 36

    c.setFont("Helvetica", 11)
    c.setFillColor(HexColor("#888888"))
    c.drawCentredString(PAGE_WIDTH / 2, y - 14, "Generated by HPGPT Agentic System")

    c.setFillColor(FOOTER_COLOR)
    c.rect(0, 0, PAGE_WIDTH, inch * 0.5, fill=1, stroke=0)
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#ffffff"))
    c.drawString(LEFT_MARGIN, inch * 0.18, "HP Confidential — Internal Use Only")


# ── Special section renderers ─────────────────────────────────────────────────

def _draw_section_heading(c, text, y, sec_type):
    if not text:
        return y
    color = (FOOTER_COLOR
             if sec_type in ("risk_section", "safety", "recommendations")
             else TITLE_COLOR)
    c.setFillColor(color)
    c.rect(LEFT_MARGIN - 6, y - 2, 3, 16, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(color)
    c.drawString(LEFT_MARGIN, y, text)
    return y - 13 - 10


def _draw_callout_block(c, text, y, bg, border, label=""):
    if not text or not text.strip():
        return y

    lines_raw = [
        l.strip() for l in text.splitlines()
        if l.strip() and not _CHART_LINE.match(l.strip())
    ]
    if not lines_raw:
        return y

    est_lines = 0
    for raw in lines_raw:
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw)
        est_lines += _estimate_wrap_count(c, content, USABLE_W - 20, "Helvetica", 11)
    box_h = est_lines * LINE_HEIGHT + 24

    if y - box_h < BOTTOM_MARGIN + 50:
        draw_footer(c)
        c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN

    c.setFillColor(bg)
    c.rect(LEFT_MARGIN, y - box_h, USABLE_W, box_h, fill=1, stroke=0)
    c.setFillColor(border)
    c.rect(LEFT_MARGIN, y - box_h, 3, box_h, fill=1, stroke=0)

    text_y = y - 14
    if label:
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(border)
        c.drawString(LEFT_MARGIN + 10, text_y, label)
        text_y -= 12

    c.setFont("Helvetica", 11)
    c.setFillColor(BODY_COLOR)
    indent = LEFT_MARGIN + 10

    for raw in lines_raw:
        is_b = bool(re.match(r'^[-•*]\s+', raw))
        is_n = bool(re.match(r'^\d+[.)]\s+', raw))
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw)
        if is_b:
            bullet_char = "•"
        elif is_n:
            m = re.match(r'^(\d+[.)]\s+)', raw)
            bullet_char = m.group(1).strip() if m else ""
        else:
            bullet_char = ""

        wrapped = _wrap_text_list(c, content, USABLE_W - 20, "Helvetica", 11)
        for j, wl in enumerate(wrapped):
            if text_y < BOTTOM_MARGIN + 30:
                break
            if j == 0 and bullet_char:
                c.drawString(indent, text_y, bullet_char)
                c.drawString(indent + 14, text_y, wl)
            else:
                c.drawString(indent + (14 if bullet_char else 0), text_y, wl)
            text_y -= LINE_HEIGHT

    return y - box_h - 10


def _draw_numbered_list(c, text, y, accent=None):
    if not text or not text.strip():
        return y
    accent = accent or TITLE_COLOR

    lines_raw = [
        l.strip() for l in text.splitlines()
        if l.strip() and not _CHART_LINE.match(l.strip())
    ]

    counter = 1
    for raw in lines_raw:
        is_b = bool(re.match(r'^[-•*]\s+', raw))
        is_n = bool(re.match(r'^\d+[.)]\s+', raw))
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw).strip()
        if not content:
            continue

        wrapped = _wrap_text_list(c, content, USABLE_W - 24, "Helvetica", 11)
        for j, wl in enumerate(wrapped):
            if y < BOTTOM_MARGIN + 50:
                draw_footer(c)
                c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN
                c.setFont("Helvetica", 11)
                c.setFillColor(BODY_COLOR)
            if j == 0 and (is_b or is_n):
                c.setFont("Helvetica-Bold", 11)
                c.setFillColor(accent)
                c.drawString(LEFT_MARGIN, y, f"{counter}.")
                c.setFont("Helvetica", 11)
                c.setFillColor(BODY_COLOR)
                c.drawString(LEFT_MARGIN + 20, y, wl)
                counter += 1
            else:
                c.setFont("Helvetica", 11)
                c.setFillColor(BODY_COLOR)
                c.drawString(LEFT_MARGIN + 20, y, wl)
            y -= LINE_HEIGHT

    return y


def _draw_checklist(c, text, y):
    if not text or not text.strip():
        return y

    lines_raw = [
        l.strip() for l in text.splitlines()
        if l.strip() and not _CHART_LINE.match(l.strip())
    ]

    for raw in lines_raw:
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw).strip()
        if not content:
            continue

        wrapped = _wrap_text_list(c, content, USABLE_W - 24, "Helvetica", 11)
        for j, wl in enumerate(wrapped):
            if y < BOTTOM_MARGIN + 50:
                draw_footer(c)
                c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN
            if j == 0:
                c.setStrokeColor(TITLE_COLOR)
                c.setFillColor(HexColor("#ffffff"))
                c.setLineWidth(0.8)
                c.rect(LEFT_MARGIN, y - 2, 9, 9, fill=1, stroke=1)
                c.setFont("Helvetica", 11)
                c.setFillColor(BODY_COLOR)
                c.drawString(LEFT_MARGIN + 14, y, wl)
            else:
                c.setFont("Helvetica", 11)
                c.setFillColor(BODY_COLOR)
                c.drawString(LEFT_MARGIN + 14, y, wl)
            y -= LINE_HEIGHT

    return y


def _draw_assessment(c, text, y):
    if not text or not text.strip():
        return y

    lines_raw = [
        l.strip() for l in text.splitlines()
        if l.strip() and not _CHART_LINE.match(l.strip())
    ]

    q_num = 0
    for raw in lines_raw:
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw).strip()
        if not content:
            continue

        is_question = bool(re.match(
            r'^(Q\d*[.:]|question\s*\d*[.:])', content, re.IGNORECASE))
        is_answer   = bool(re.match(
            r'^(A\d*[.:]|ans(wer)?\s*[.:])', content, re.IGNORECASE))

        if is_question:
            q_num += 1
            body  = re.sub(r'^(Q\d*[.:]|question\s*\d*[.:]\s*)', '',
                           content, flags=re.IGNORECASE).strip()
            wrapped = _wrap_text_list(c, body, USABLE_W - 24, "Helvetica-Bold", 11)
            for j, wl in enumerate(wrapped):
                if y < BOTTOM_MARGIN + 50:
                    draw_footer(c); c.showPage(); y = PAGE_HEIGHT - TOP_MARGIN
                if j == 0:
                    c.setFont("Helvetica-Bold", 11)
                    c.setFillColor(TITLE_COLOR)
                    c.drawString(LEFT_MARGIN, y, f"Q{q_num}.")
                    c.drawString(LEFT_MARGIN + 22, y, wl)
                else:
                    c.drawString(LEFT_MARGIN + 22, y, wl)
                y -= LINE_HEIGHT
        elif is_answer:
            body = re.sub(r'^(A\d*[.:]|ans(wer)?\s*[.:]\s*)', '',
                          content, flags=re.IGNORECASE).strip()
            wrapped = _wrap_text_list(c, body, USABLE_W - 30, "Helvetica", 10)
            for j, wl in enumerate(wrapped):
                if y < BOTTOM_MARGIN + 50:
                    draw_footer(c); c.showPage(); y = PAGE_HEIGHT - TOP_MARGIN
                if j == 0:
                    c.setFont("Helvetica-Oblique", 10)
                    c.setFillColor(HexColor("#555555"))
                    c.drawString(LEFT_MARGIN + 14, y, f"→ {wl}")
                else:
                    c.drawString(LEFT_MARGIN + 22, y, wl)
                y -= LINE_HEIGHT - 2
        else:
            y = draw_body(c, raw, y)

    return y


def draw_kpi_table(c, td, y):
    if not td or len(td) < 2:
        return y
    return draw_table(c, td, y)


# ── Helpers ───────────────────────────────────────────────────────────────────

STOP = {'by', 'and', 'the', 'of', 'in', 'a', 'an', 'to', 'for', 'from',
        'with', 'data', 'analysis', 'report', 'file', 'product', 'wise',
        'per', 'vs', 'are', 'is'}

def _keywords(text):
    return set(re.findall(r'\w+', text.lower())) - STOP

ABBREVS = {'HPCL', 'BPCL', 'IOCL', 'LPG', 'ATF', 'KL', 'MT', 'OMC', 'HP', 'INR'}

def _clean_col_name(name):
    name  = re.sub(r'—.*$', '', name).strip()
    name  = name.replace('_KL', '').replace('_MT', '').replace('_INR', '')
    parts = name.replace('_', ' ').split()
    return ' '.join(
        p if p.upper() in ABBREVS else p.title() for p in parts
    )


def _estimate_wrap_count(c, text, max_w, font, size):
    words = text.split()
    count, cur = 1, ""
    for w in words:
        t = f"{cur} {w}".strip()
        if c.stringWidth(t, font, size) <= max_w:
            cur = t
        else:
            if cur: count += 1
            cur = w
    return max(count, 1)


def _wrap_text_list(c, text, max_w, font="Helvetica", size=11):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if c.stringWidth(t, font, size) <= max_w:
            cur = t
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    return lines or ['']


# ── Chart-to-section matching ──────────────────────────────────────────────────

def _section_keywords(sec):
    heading_kw = _keywords(sec.get('heading', ''))
    table_kw   = set()
    if sec.get('table'):
        for row in sec['table']:
            for cell in row:
                table_kw |= _keywords(str(cell))
    body_kw = _keywords(sec.get('body', ''))
    return heading_kw, table_kw, body_kw


def _assign_charts_to_sections(sections, chart_images):
    section_kw  = [_section_keywords(sec) for sec in sections]
    assignments = {}
    unplaced    = []

    for idx, chart in enumerate(chart_images):
        if isinstance(chart, dict):
            ct = chart.get("title", "")
            img_path = chart.get("path", "")
            col = chart.get("column", "")
        else:
            ct = chart[0]
            img_path = chart[1]
            col = chart[2] if len(chart) > 2 else ""
        
        if not os.path.exists(img_path):
            continue
        chart_kw = _keywords(ct)
        if col:
            chart_kw = chart_kw | _keywords(col)
            
        if not chart_kw:
            unplaced.append(idx)
            continue

        best_sec, best_score = None, 0
        for sec_idx, (heading_kw, table_kw, body_kw) in enumerate(section_kw):
            score = (3 * len(chart_kw & heading_kw)
                     + 2 * len(chart_kw & table_kw)
                     + 1 * len(chart_kw & body_kw))
            if score > best_score:
                best_score, best_sec = score, sec_idx

        if best_sec is not None and best_score > 0:
            assignments.setdefault(best_sec, []).append(idx)
        else:
            unplaced.append(idx)

    return assignments, unplaced


def _img_h(path, max_h=CHART_H):
    try:
        from PIL import Image as PI
        img = PI.open(path)
        iw, ih = img.size
        scale  = min(USABLE_W / iw, max_h / ih, 1.0)
        return ih * scale
    except Exception:
        return max_h


def _place_chart(c, img_path, y):
    img_h = _img_h(img_path)
    if y - img_h - 10 < BOTTOM_MARGIN + 50:
        draw_footer(c)
        c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN
    return draw_image(c, img_path, y)


def clean_text(text):
    lines, in_table = [], False
    for line in text.splitlines():
        s = line.strip()
        if _CHART_LINE.match(s):
            continue
        if s.startswith('|') and s.endswith('|'):
            in_table = True; lines.append(line); continue
        if in_table:
            lines.append(line)
            if s == '': in_table = False
            continue
        m = re.match(r'^(#{1,3})\s+(.*)$', line)
        if m:
            lines.append(
                f"{m.group(1)} {re.sub(r'[*_]+', '', m.group(2)).strip()}")
            continue
        if re.match(r'^[=\-_*]{3,}$', s):
            continue
        lines.append(re.sub(r'[*_]+', '', line))
    return "\n".join(lines)


def wrap_text(c, text, max_w, font="Helvetica", size=11):
    return _wrap_text_list(c, text, max_w, font, size)


# ── Drawing primitives ────────────────────────────────────────────────────────

def _draw_heading(c, text, y, size, color, underline=False):
    if not text:
        return y
    c.setFont("Helvetica-Bold", size)
    c.setFillColor(color)
    c.drawString(LEFT_MARGIN, y, text)
    if underline:
        c.setStrokeColor(FOOTER_COLOR)
        c.setLineWidth(1.2)
        c.line(LEFT_MARGIN, y - 4, PAGE_WIDTH - RIGHT_MARGIN, y - 4)
    return y - size - 12


def draw_body(c, text, y):
    c.setFont("Helvetica", 11)
    c.setFillColor(BODY_COLOR)
    for line in text.splitlines():
        s = line.strip()
        if not s:
            y -= 6; continue
        if _CHART_LINE.match(s):
            continue

        is_b = bool(re.match(r'^[-•+\*]\s+', s))
        is_n = bool(re.match(r'^\d+[.)]\s+', s))
        if is_b:
            content, bullet, indent = re.sub(r'^[-•+\*]\s+', '', s), '•', LEFT_MARGIN + 16
        elif is_n:
            m = re.match(r'^(\d+[.)]\s+)(.*)', s)
            bullet  = m.group(1).strip() if m else ''
            content = m.group(2) if m else s
            indent  = LEFT_MARGIN + 20
        else:
            content, bullet, indent = s, '', LEFT_MARGIN

        wrapped = _wrap_text_list(c, content, USABLE_W - (indent - LEFT_MARGIN))
        for j, wl in enumerate(wrapped):
            if y < BOTTOM_MARGIN + 50:
                draw_footer(c)
                c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN
                c.setFont("Helvetica", 11)
                c.setFillColor(BODY_COLOR)
            if j == 0 and bullet:
                c.drawString(LEFT_MARGIN, y, bullet)
            c.drawString(indent, y, wl)
            y -= LINE_HEIGHT
    return y


def draw_table(c, td, y):
    if not td or len(td) < 2:
        return y
    cols = max(len(r) for r in td)
    cw, rh, fs = USABLE_W / cols, 18, 9
    if y - len(td) * rh < BOTTOM_MARGIN + 50:
        draw_footer(c); c.showPage(); y = PAGE_HEIGHT - TOP_MARGIN
    for ri, row in enumerate(td):
        while len(row) < cols:
            row.append("")
        ry = y - ri * rh
        for ci, cell in enumerate(row):
            cx = LEFT_MARGIN + ci * cw
            bg = (TABLE_HDR_BG if ri == 0
                  else (TABLE_ALT_BG if ri % 2 == 0 else HexColor("#ffffff")))
            c.setFillColor(bg)
            c.rect(cx, ry - rh + 4, cw, rh, fill=1, stroke=0)
            c.setStrokeColor(HexColor("#cccccc")); c.setLineWidth(0.3)
            c.rect(cx, ry - rh + 4, cw, rh, fill=0, stroke=1)
            c.setFillColor(HexColor("#ffffff") if ri == 0 else BODY_COLOR)
            c.setFont("Helvetica-Bold" if ri == 0 else "Helvetica", fs)
            txt = str(cell)
            while txt and c.stringWidth(txt, "Helvetica", fs) > cw - 10:
                txt = txt[:-1]
            c.drawString(cx + 5, ry - rh + 4 + (rh - fs) / 2, txt)
    return y - len(td) * rh - 14


def draw_image(c, path, y, max_h=CHART_H):
    try:
        from PIL import Image as PI
        img    = PI.open(path)
        iw, ih = img.size
        sc     = min(USABLE_W / iw, max_h / ih, 1.0)
        dw, dh = iw * sc, ih * sc
        if y - dh - 10 < BOTTOM_MARGIN + 50:
            draw_footer(c); c.showPage(); y = PAGE_HEIGHT - TOP_MARGIN
        c.drawImage(path, (PAGE_WIDTH - dw) / 2, y - dh, width=dw, height=dh)
        c.setStrokeColor(HexColor("#eeeeee")); c.setLineWidth(0.5)
        c.rect((PAGE_WIDTH - dw) / 2, y - dh, dw, dh, fill=0, stroke=1)
        y -= dh + 10
    except Exception as e:
        logging.warning(f"Image embed failed {path}: {e}")
    return y


def draw_footer(c):
    c.saveState()
    c.setStrokeColor(FOOTER_COLOR); c.setLineWidth(0.8)
    c.line(LEFT_MARGIN, inch * 0.65, PAGE_WIDTH - RIGHT_MARGIN, inch * 0.65)
    c.setFont("Helvetica-Bold", 9); c.setFillColor(FOOTER_COLOR)
    c.drawString(LEFT_MARGIN, inch * 0.45, "HPGPT")
    pg = f"Page {c.getPageNumber()}"
    c.drawString(PAGE_WIDTH - RIGHT_MARGIN
                 - c.stringWidth(pg, "Helvetica-Bold", 9),
                 inch * 0.45, pg)
    c.restoreState()
