import os
import re
import logging

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

# Accent colors for callout boxes (used by special section types)
CALLOUT_BG   = HexColor("#eef1f8")
CALLOUT_BORDER = HexColor("#00205b")
WARNING_BG   = HexColor("#fff4f4")
WARNING_BORDER = HexColor("#e4002b")

USABLE_W = PAGE_WIDTH  - LEFT_MARGIN - RIGHT_MARGIN
USABLE_H = PAGE_HEIGHT - TOP_MARGIN  - BOTTOM_MARGIN

# Regex to detect raw chart placeholder lines
_CHART_LINE = re.compile(r'^\s*\[CHART:', re.IGNORECASE)

# ── Section-type detection ─────────────────────────────────────────────────────
# Certain section headings get special visual treatment:
#   - executive_summary  → blue left-border callout
#   - risk / warning     → red left-border callout
#   - kpi_dashboard      → rendered as a metric table
#   - recommendations    → numbered list with red bullet

_SECTION_TYPE_MAP = [
    (re.compile(r'\bexecutive\s+summary\b', re.IGNORECASE),       "executive_summary"),
    (re.compile(r'\brisk\b|\bwarning\b|\bcaution\b', re.IGNORECASE), "risk_section"),
    (re.compile(r'\bkpi\b|\bdashboard\b|\bscorecard\b', re.IGNORECASE), "kpi_section"),
    (re.compile(r'\brecommendations?\b', re.IGNORECASE),           "recommendations"),
    (re.compile(r'\blearning\s+objectives?\b', re.IGNORECASE),     "objectives"),
    (re.compile(r'\bcase\s+study\b', re.IGNORECASE),               "case_study"),
    (re.compile(r'\bassessment\b|\bquiz\b', re.IGNORECASE),        "assessment"),
    (re.compile(r'\bsafety\b|\bwarning\b|\bstop\b', re.IGNORECASE), "safety"),
    (re.compile(r'\bchecklist\b', re.IGNORECASE),                  "checklist"),
]


def _detect_section_type(heading: str) -> str:
    for pattern, stype in _SECTION_TYPE_MAP:
        if pattern.search(heading):
            return stype
    return "body"


def generate_pdf(body, output_path="outputs/output.pdf",
                 title="AI Generated PDF", chart_images=None):
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True
    )
    chart_images = _dedupe_charts(list(chart_images or []))

    display_title = _clean_title(title)

    # Strip chart placeholders from body before processing
    body = _strip_chart_placeholders(body)

    c = canvas.Canvas(output_path, pagesize=LETTER)
    _draw_cover(c, display_title)
    c.showPage()

    y = PAGE_HEIGHT - TOP_MARGIN
    sections = parse_sections(clean_text(body))

    # Assign each chart to its best-matching section
    assignments, unplaced_idx = _assign_charts_to_sections(sections, chart_images)

    for sec_idx, sec in enumerate(sections):
        level, heading, content, table = \
            sec['level'], sec['heading'], sec['body'], sec['table']

        if y < BOTTOM_MARGIN + 120:
            draw_footer(c); c.showPage()
            y = PAGE_HEIGHT - TOP_MARGIN

        sec_type = _detect_section_type(heading) if heading else "body"

        # ── Heading rendering ─────────────────────────────────────────────────
        if level == 1:
            y = _draw_heading(c, heading, y, 15, TITLE_COLOR, underline=True)
        elif level == 2:
            y = _draw_section_heading(c, heading, y, sec_type)
        elif level == 3:
            y = _draw_heading(c, heading, y, 11, HexColor("#333333"))

        # ── Content rendering — section-type-aware ────────────────────────────
        if sec_type == "executive_summary":
            y = _draw_callout_block(c, content, y,
                                    bg=CALLOUT_BG, border=CALLOUT_BORDER,
                                    label="EXECUTIVE SUMMARY")
        elif sec_type == "risk_section" or sec_type == "safety":
            y = _draw_callout_block(c, content, y,
                                    bg=WARNING_BG, border=WARNING_BORDER,
                                    label="⚠ ATTENTION")
        elif sec_type == "kpi_section" and table:
            y = draw_kpi_table(c, table, y)
        elif sec_type == "checklist":
            y = _draw_checklist(c, content, y)
        elif sec_type == "objectives":
            y = _draw_numbered_list(c, content, y, accent=TITLE_COLOR)
        elif sec_type == "recommendations":
            y = _draw_numbered_list(c, content, y, accent=FOOTER_COLOR)
        elif sec_type == "assessment":
            y = _draw_assessment(c, content, y)
        elif table:
            y = draw_table(c, table, y)
        else:
            y = draw_body(c, content, y)

        # ── Inline charts ─────────────────────────────────────────────────────
        for idx in assignments.get(sec_idx, []):
            _, img_path = chart_images[idx]
            y = _place_chart(c, img_path, y)

        y -= 8

    # ── Appendix: unplaced charts ─────────────────────────────────────────────
    if unplaced_idx:
        draw_footer(c); c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN
        y = _draw_heading(c, "Charts & Visualisations", y, 15, TITLE_COLOR, underline=True)
        y -= 8

        for idx in unplaced_idx:
            ct, img_path = chart_images[idx]
            label   = _clean_col_name(ct)
            img_h   = _img_h(img_path)
            needed  = img_h + 26

            if y - needed < BOTTOM_MARGIN + 50:
                draw_footer(c); c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN

            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(TITLE_COLOR)
            c.drawString(LEFT_MARGIN, y, label)
            y -= 14
            y  = draw_image(c, img_path, y)
            y -= 14

    draw_footer(c)
    c.save()
    logging.info(f"PDF saved to: {output_path}")
    print(f"PDF saved to: {output_path}")


# ── Title cleaning ─────────────────────────────────────────────────────────────

def _clean_title(title):
    t = re.sub(r'^[#*\-_=\s"\']+|[#*\-_=\s"\']+$', '', title).strip()
    skip = {'a', 'an', 'the', 'of', 'in', 'on', 'with', 'and', 'or', 'for', 'to', 'by',
            'give', 'me', 'make', 'create', 'generate', 'write', 'build', 'produce', 'get'}
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


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedupe_charts(chart_images):
    seen_paths  = set()
    seen_titles = set()
    result = []
    for ct, path in chart_images:
        norm_title = ct.lower().strip()
        if path in seen_paths or norm_title in seen_titles:
            continue
        seen_paths.add(path)
        seen_titles.add(norm_title)
        result.append((ct, path))
    return result


# ── Cover ─────────────────────────────────────────────────────────────────────

def _draw_cover(c, title):
    # Header band
    c.setFillColor(TITLE_COLOR)
    c.rect(0, PAGE_HEIGHT - inch*1.8, PAGE_WIDTH, inch*1.8, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(HexColor("#ffffff"))
    c.drawString(LEFT_MARGIN, PAGE_HEIGHT - inch*0.9, "HPGPT")
    c.setFillColor(FOOTER_COLOR)
    c.rect(0, PAGE_HEIGHT - inch*2.0, PAGE_WIDTH, inch*0.2, fill=1, stroke=0)

    # Title
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
        c.drawCentredString(PAGE_WIDTH/2, y, line)
        y -= 36

    c.setFont("Helvetica", 11)
    c.setFillColor(HexColor("#888888"))
    c.drawCentredString(PAGE_WIDTH/2, y - 14, "Generated by HPGPT Agentic System")

    # Footer band
    c.setFillColor(FOOTER_COLOR)
    c.rect(0, 0, PAGE_WIDTH, inch*0.5, fill=1, stroke=0)
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#ffffff"))
    c.drawString(LEFT_MARGIN, inch*0.18, "HP Confidential — Internal Use Only")


# ── Special section renderers ─────────────────────────────────────────────────

def _draw_section_heading(c, text, y, sec_type):
    """Level-2 heading with optional left-border accent for special sections."""
    if not text:
        return y

    color = FOOTER_COLOR if sec_type in ("risk_section", "safety", "recommendations") else TITLE_COLOR

    # Accent bar on the left
    c.setFillColor(color)
    c.rect(LEFT_MARGIN - 6, y - 2, 3, 16, fill=1, stroke=0)

    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(color)
    c.drawString(LEFT_MARGIN, y, text)
    return y - 13 - 10


def _draw_callout_block(c, text, y, bg, border, label=""):
    """
    Render a section body inside a shaded callout box with a coloured border.
    Used for Executive Summary, Risk sections, Safety warnings.
    """
    if not text or not text.strip():
        return y

    lines_raw = [l.strip() for l in text.splitlines() if l.strip()
                 and not _CHART_LINE.match(l.strip())]
    if not lines_raw:
        return y

    # Estimate box height
    est_lines = 0
    for raw in lines_raw:
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw)
        wrapped = _estimate_wrap_count(c, content, USABLE_W - 20, "Helvetica", 11)
        est_lines += wrapped
    box_h = est_lines * LINE_HEIGHT + 24  # 12px padding top+bottom

    if y - box_h < BOTTOM_MARGIN + 50:
        draw_footer(c); c.showPage()
        y = PAGE_HEIGHT - TOP_MARGIN

    # Box background
    c.setFillColor(bg)
    c.rect(LEFT_MARGIN, y - box_h, USABLE_W, box_h, fill=1, stroke=0)

    # Left border accent
    c.setFillColor(border)
    c.rect(LEFT_MARGIN, y - box_h, 3, box_h, fill=1, stroke=0)

    # Optional label
    text_y = y - 14
    if label:
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(border)
        c.drawString(LEFT_MARGIN + 10, text_y, label)
        text_y -= 12

    # Body text
    c.setFont("Helvetica", 11)
    c.setFillColor(BODY_COLOR)
    indent = LEFT_MARGIN + 10

    for raw in lines_raw:
        is_b = bool(re.match(r'^[-•*]\s+', raw))
        is_n = bool(re.match(r'^\d+[.)]\s+', raw))
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw)
        bullet_char = "•" if is_b else (re.match(r'^(\d+[.)]\s+)', raw).group(1).strip() if is_n else "")

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
    """
    Render body text as a visually distinct numbered list with accent color
    for the number. Used for Learning Objectives and Recommendations.
    """
    if not text or not text.strip():
        return y
    accent = accent or TITLE_COLOR

    lines_raw = [l.strip() for l in text.splitlines() if l.strip()
                 and not _CHART_LINE.match(l.strip())]

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
                draw_footer(c); c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN
                c.setFont("Helvetica", 11); c.setFillColor(BODY_COLOR)

            if j == 0 and (is_b or is_n):
                c.setFont("Helvetica-Bold", 11)
                c.setFillColor(accent)
                num_str = f"{counter}."
                c.drawString(LEFT_MARGIN, y, num_str)
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
    """
    Render bullet points as a checkbox checklist (□ item).
    Used for Compliance Checklist and similar sections.
    """
    if not text or not text.strip():
        return y

    lines_raw = [l.strip() for l in text.splitlines() if l.strip()
                 and not _CHART_LINE.match(l.strip())]

    for raw in lines_raw:
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw).strip()
        if not content:
            continue

        wrapped = _wrap_text_list(c, content, USABLE_W - 24, "Helvetica", 11)
        for j, wl in enumerate(wrapped):
            if y < BOTTOM_MARGIN + 50:
                draw_footer(c); c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN

            if j == 0:
                # Draw checkbox
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
    """
    Render assessment/quiz questions with visual Q/A distinction.
    Questions are bold blue; answers are indented grey.
    """
    if not text or not text.strip():
        return y

    lines_raw = [l.strip() for l in text.splitlines() if l.strip()
                 and not _CHART_LINE.match(l.strip())]

    q_num = 0
    for raw in lines_raw:
        content = re.sub(r'^[-•*+\d.)]+\s*', '', raw).strip()
        if not content:
            continue

        # Heuristic: lines starting with Q or numbered are questions;
        # lines starting with A or "Answer" are answers.
        is_question = bool(re.match(r'^(Q\d*[.:]|question\s*\d*[.:])', content, re.IGNORECASE))
        is_answer   = bool(re.match(r'^(A\d*[.:]|ans(wer)?\s*[.:])', content, re.IGNORECASE))

        if is_question:
            q_num += 1
            label = f"Q{q_num}."
            body  = re.sub(r'^(Q\d*[.:]|question\s*\d*[.:]\s*)', '', content, flags=re.IGNORECASE).strip()
            wrapped = _wrap_text_list(c, body, USABLE_W - 24, "Helvetica-Bold", 11)
            for j, wl in enumerate(wrapped):
                if y < BOTTOM_MARGIN + 50:
                    draw_footer(c); c.showPage(); y = PAGE_HEIGHT - TOP_MARGIN
                if j == 0:
                    c.setFont("Helvetica-Bold", 11); c.setFillColor(TITLE_COLOR)
                    c.drawString(LEFT_MARGIN, y, label)
                    c.drawString(LEFT_MARGIN + 22, y, wl)
                else:
                    c.drawString(LEFT_MARGIN + 22, y, wl)
                y -= LINE_HEIGHT
        elif is_answer:
            body = re.sub(r'^(A\d*[.:]|ans(wer)?\s*[.:]\s*)', '', content, flags=re.IGNORECASE).strip()
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
            # Fallback: render as regular body
            y = draw_body(c, raw, y)

    return y


def draw_kpi_table(c, td, y):
    """
    Render a KPI dashboard as a visually enhanced table with metric tiles.
    Falls back to regular table rendering if data is too complex.
    """
    if not td or len(td) < 2:
        return y

    # Use regular table for complex multi-column KPI data
    return draw_table(c, td, y)


# ── Helpers ───────────────────────────────────────────────────────────────────

STOP = {'by','and','the','of','in','a','an','to','for','from','with','data',
        'analysis','report','file','product','wise','per','vs','are','is'}

def _keywords(text):
    return set(re.findall(r'\w+', text.lower())) - STOP

ABBREVS = {'HPCL','BPCL','IOCL','LPG','ATF','KL','MT','OMC','HP','INR'}

def _clean_col_name(name):
    name = re.sub(r'—.*$', '', name).strip()
    name = name.replace('_KL','').replace('_MT','').replace('_INR','')
    parts = name.replace('_',' ').split()
    return ' '.join(p if p.upper() in ABBREVS else p.title() for p in parts)


def _estimate_wrap_count(c, text, max_w, font, size):
    """Estimate how many wrapped lines a piece of text will occupy."""
    words = text.split()
    count, cur = 1, ""
    for w in words:
        t = f"{cur} {w}".strip()
        if c.stringWidth(t, font, size) <= max_w:
            cur = t
        else:
            if cur:
                count += 1
            cur = w
    return max(count, 1)


def _wrap_text_list(c, text, max_w, font="Helvetica", size=11):
    """Wrap text and return list of line strings."""
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

    for idx, (ct, img_path) in enumerate(chart_images):
        if not os.path.exists(img_path):
            continue

        chart_kw = _keywords(ct)
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
        scale = min(USABLE_W/iw, max_h/ih, 1.0)
        return ih * scale
    except Exception:
        return max_h


def _place_chart(c, img_path, y):
    img_h = _img_h(img_path)
    if y - img_h - 10 < BOTTOM_MARGIN + 50:
        draw_footer(c); c.showPage()
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
            lines.append(f"{m.group(1)} {re.sub(r'[*_]+','',m.group(2)).strip()}")
            continue
        if re.match(r'^[=\-_*]{3,}$', s): continue
        lines.append(re.sub(r'[*_]+', '', line))
    return "\n".join(lines)


def parse_sections(text):
    pat = re.compile(r'^(#{1,3})\s+(.*)$', re.MULTILINE)
    ms  = list(pat.finditer(text))
    if not ms:
        return [{'level':0,'heading':'','body':text.strip(),'table':None}]
    secs = []
    if ms[0].start() > 0:
        pre = text[:ms[0].start()].strip()
        if pre: secs.append({'level':0,'heading':'','body':pre,'table':None})
    for i, m in enumerate(ms):
        body = text[m.end(): ms[i+1].start() if i+1<len(ms) else len(text)].strip()
        secs.append({'level':len(m.group(1)),'heading':m.group(2).strip(),
                     'body':body,'table':extract_table(body)})
    return secs


def extract_table(text):
    lines = [l.strip() for l in text.splitlines() if '|' in l]
    if len(lines) < 2: return None
    rows = [l for l in lines if not re.match(r'^[\|\s\-:]+$', l)]
    if len(rows) < 2: return None
    return [[c.strip() for c in r.split('|') if c.strip()] for r in rows]


def wrap_text(c, text, max_w, font="Helvetica", size=11):
    return _wrap_text_list(c, text, max_w, font, size)


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_heading(c, text, y, size, color, underline=False):
    if not text: return y
    c.setFont("Helvetica-Bold", size)
    c.setFillColor(color)
    c.drawString(LEFT_MARGIN, y, text)
    if underline:
        c.setStrokeColor(FOOTER_COLOR); c.setLineWidth(1.2)
        c.line(LEFT_MARGIN, y-4, PAGE_WIDTH-RIGHT_MARGIN, y-4)
    return y - size - 12


def draw_body(c, text, y):
    c.setFont("Helvetica", 11)
    c.setFillColor(BODY_COLOR)
    for line in text.splitlines():
        s = line.strip()
        if not s: y -= 6; continue
        if _CHART_LINE.match(s): continue

        is_b = bool(re.match(r'^[-•+\*]\s+', s))
        is_n = bool(re.match(r'^\d+[.)]\s+', s))
        if is_b:
            content, bullet, indent = re.sub(r'^[-•+\*]\s+','',s), '•', LEFT_MARGIN+16
        elif is_n:
            m = re.match(r'^(\d+[.)]\s+)(.*)', s)
            bullet = m.group(1).strip() if m else ''
            content = m.group(2) if m else s
            indent = LEFT_MARGIN+20
        else:
            content, bullet, indent = s, '', LEFT_MARGIN

        wrapped = _wrap_text_list(c, content, USABLE_W-(indent-LEFT_MARGIN))
        for j, wl in enumerate(wrapped):
            if y < BOTTOM_MARGIN + 50:
                draw_footer(c); c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN
                c.setFont("Helvetica", 11); c.setFillColor(BODY_COLOR)
            if j == 0 and bullet:
                c.drawString(LEFT_MARGIN, y, bullet)
            c.drawString(indent, y, wl)
            y -= LINE_HEIGHT
    return y


def draw_table(c, td, y):
    if not td or len(td) < 2: return y
    cols = max(len(r) for r in td)
    cw   = USABLE_W/cols; rh = 18; fs = 9
    if y - len(td)*rh < BOTTOM_MARGIN+50:
        draw_footer(c); c.showPage(); y = PAGE_HEIGHT - TOP_MARGIN
    for ri, row in enumerate(td):
        while len(row) < cols: row.append("")
        ry = y - ri*rh
        for ci, cell in enumerate(row):
            cx = LEFT_MARGIN + ci*cw
            bg = TABLE_HDR_BG if ri==0 else (TABLE_ALT_BG if ri%2==0 else HexColor("#ffffff"))
            c.setFillColor(bg)
            c.rect(cx, ry-rh+4, cw, rh, fill=1, stroke=0)
            c.setStrokeColor(HexColor("#cccccc")); c.setLineWidth(0.3)
            c.rect(cx, ry-rh+4, cw, rh, fill=0, stroke=1)
            c.setFillColor(HexColor("#ffffff") if ri==0 else BODY_COLOR)
            c.setFont("Helvetica-Bold" if ri==0 else "Helvetica", fs)
            txt = str(cell)
            while txt and c.stringWidth(txt,"Helvetica",fs) > cw-10: txt = txt[:-1]
            c.drawString(cx+5, ry-rh+4+(rh-fs)/2, txt)
    return y - len(td)*rh - 14


def draw_image(c, path, y, max_h=CHART_H):
    try:
        from PIL import Image as PI
        img = PI.open(path)
        iw, ih = img.size
        sc = min(USABLE_W/iw, max_h/ih, 1.0)
        dw, dh = iw*sc, ih*sc
        if y - dh - 10 < BOTTOM_MARGIN + 50:
            draw_footer(c); c.showPage(); y = PAGE_HEIGHT - TOP_MARGIN
        c.drawImage(path, (PAGE_WIDTH-dw)/2, y-dh, width=dw, height=dh)
        c.setStrokeColor(HexColor("#eeeeee")); c.setLineWidth(0.5)
        c.rect((PAGE_WIDTH-dw)/2, y-dh, dw, dh, fill=0, stroke=1)
        y -= dh + 10
    except Exception as e:
        logging.warning(f"Image embed failed {path}: {e}")
    return y


def draw_footer(c):
    c.saveState()
    c.setStrokeColor(FOOTER_COLOR); c.setLineWidth(0.8)
    c.line(LEFT_MARGIN, inch*0.65, PAGE_WIDTH-RIGHT_MARGIN, inch*0.65)
    c.setFont("Helvetica-Bold", 9); c.setFillColor(FOOTER_COLOR)
    c.drawString(LEFT_MARGIN, inch*0.45, "HPGPT")
    pg = f"Page {c.getPageNumber()}"
    c.drawString(PAGE_WIDTH-RIGHT_MARGIN-c.stringWidth(pg,"Helvetica-Bold",9), inch*0.45, pg)
    c.restoreState()