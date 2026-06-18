import os
import re
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
 
TITLE_COLOR  = RGBColor(0, 32, 91)
BODY_COLOR   = RGBColor(0, 0, 0)
FOOTER_COLOR = RGBColor(0xE4, 0x00, 0x2B)
 
# Regex to detect raw chart placeholders the LLM sometimes emits
_CHART_LINE = re.compile(r'^\s*\[CHART:', re.IGNORECASE)
 
 
def generate_docx(body, output_path, title="Generated Document", chart_images=None):
    chart_images = _dedupe_charts(list(chart_images or []))
    doc = Document()
 
    # clean title — strip raw prompt artifacts and quotes
    display_title = _clean_title(title)
 
    if display_title:
        para = doc.add_paragraph()
        run = para.add_run(display_title)
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = TITLE_COLOR
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.space_after = Pt(12)
 
    # strip chart placeholder lines before parsing
    body = _strip_chart_placeholders(body)
 
    sections = parse_sections(clean_text(body))
 
    # FIX 3: assign each chart to the section it is actually about, scoring
    # against heading + table content + body text (not a heading substring
    # match). Chart titles are usually short data labels (e.g. "Hpcl Kl")
    # that don't appear inside section headings (e.g. "Sales Performance")
    # but DO overlap with the table/prose the chart was generated from. Only
    # charts with zero overlap anywhere fall back to the appendix.
    assignments, unplaced_idx = _assign_charts_to_sections(sections, chart_images)
 
    for sec_idx, sec in enumerate(sections):
        level, heading, content, table = sec['level'], sec['heading'], sec['body'], sec['table']
 
        if level == 1:
            _add_heading(doc, heading, 15, TITLE_COLOR)
        elif level == 2:
            _add_heading(doc, heading, 13, TITLE_COLOR)
        elif level == 3:
            _add_heading(doc, heading, 11, BODY_COLOR)
 
        if table:
            _add_table(doc, table)
        else:
            _add_body_content(doc, content)
 
        for idx in assignments.get(sec_idx, []):
            _, img_path = chart_images[idx]
            _add_image(doc, img_path)
 
    if unplaced_idx:
        para = doc.add_paragraph()
        run = para.add_run("Charts & Visualisations")
        run.font.size = Pt(14)
        run.font.bold = True
        run.font.color.rgb = TITLE_COLOR
        para.space_after = Pt(8)
 
        for idx in unplaced_idx:
            _, img_path = chart_images[idx]
            _add_image(doc, img_path)
 
    _add_footer(doc)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    doc.save(output_path)
    print(f"DOCX saved to: {output_path}")
 
 
# ── Title cleaning ─────────────────────────────────────────────────────────────
 
def _clean_title(title):
    """
    Convert a raw user prompt used as title into a proper document title.
    e.g. "give me a hr document" → "Human Resources Document"
    Strips markdown symbols, quotes, and leading/trailing junk.
    """
    t = re.sub(r'^[#*\-_=\s"\']+|[#*\-_=\s"\']+$', '', title).strip()
    # Title-case it so it looks like a proper document title
    skip = {'a','an','the','of','in','on','with','and','or','for','to','by'}
    words = t.split()
    result = []
    for i, w in enumerate(words):
        if i == 0 or i == len(words) - 1 or w.lower() not in skip:
            result.append(w.capitalize())
        else:
            result.append(w.lower())
    return ' '.join(result)
 
 
# ── Chart placeholder stripping ───────────────────────────────────────────────
 
def _strip_chart_placeholders(text):
    """
    Remove lines like [CHART: bar title=... labels=... values=...]
    so they never appear as body text in the output document.
    """
    cleaned = []
    for line in text.splitlines():
        if _CHART_LINE.match(line):
            continue
        cleaned.append(line)
    return '\n'.join(cleaned)
 
 
# ── Deduplication ─────────────────────────────────────────────────────────────
 
def _dedupe_charts(chart_images):
    """
    Same dedup logic as pdf_writer: drop charts that repeat either an image
    path or a (case-insensitive) title, so the same chart can't get embedded
    twice (e.g. once via section matching and once via the appendix).
    """
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
 
 
# ── Text processing ───────────────────────────────────────────────────────────
 
def clean_text(text):
    lines, in_table = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            lines.append(line)
            continue
        if in_table:
            lines.append(line)
            if stripped == '':
                in_table = False
            continue
        m = re.match(r'^(#{1,3})\s+(.*)$', line)
        if m:
            content = re.sub(r'[*_]+', '', m.group(2)).strip()
            lines.append(f"{m.group(1)} {content}")
            continue
        if re.match(r'^[=\-_*]{3,}$', stripped):
            continue
        lines.append(re.sub(r'[*_]+', '', line))
    return "\n".join(lines)
 
 
def parse_sections(text):
    pattern = re.compile(r'^(#{1,3})\s+(.*)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
 
    if not matches:
        return [{'level': 0, 'heading': '', 'body': text.strip(), 'table': None}]
 
    sections = []
    if matches[0].start() > 0:
        pre = text[:matches[0].start()].strip()
        if pre:
            sections.append({'level': 0, 'heading': '', 'body': pre, 'table': None})
 
    for i, m in enumerate(matches):
        body = text[m.end(): matches[i+1].start() if i+1 < len(matches) else len(text)].strip()
        sections.append({
            'level':   len(m.group(1)),
            'heading': m.group(2).strip(),
            'body':    body,
            'table':   _extract_table(body)
        })
    return sections
 
 
def _extract_table(text):
    lines = [l.strip() for l in text.splitlines() if '|' in l]
    if len(lines) < 2:
        return None
    rows = [l for l in lines if not re.match(r'^[\|\s\-:]+$', l)]
    if len(rows) < 2:
        return None
    return [[c.strip() for c in r.split('|') if c.strip()] for r in rows]
 
 
# ── Chart-to-section matching ──────────────────────────────────────────────────
 
STOP = {'by','and','the','of','in','a','an','to','for','from','with','data',
        'analysis','report','file','product','wise','per','vs','are','is'}
 
def _keywords(text):
    return set(re.findall(r'\w+', text.lower())) - STOP
 
 
def _section_keywords(sec):
    """
    Build the keyword vocabulary describing what a section is "about":
    its heading, its table headers/cells, and its body text.
 
    Chart titles are usually short data labels (e.g. "Hpcl Kl") rather than
    English phrases, so they rarely overlap with heading words like "Sales
    Performance". They DO usually overlap with the table the chart was built
    from (company names, units, column headers) — table content is the
    strongest signal. Heading is checked too in case a title happens to echo
    it, and body text is a lighter-weight fallback for narrative mentions.
    """
    heading_kw = _keywords(sec.get('heading', ''))
 
    table_kw = set()
    if sec.get('table'):
        for row in sec['table']:
            for cell in row:
                table_kw |= _keywords(str(cell))
 
    body_kw = _keywords(sec.get('body', ''))
 
    return heading_kw, table_kw, body_kw
 
 
def _assign_charts_to_sections(sections, chart_images):
    """
    Score every chart against every section and assign it to its best match.
 
    Returns:
        assignments: {section_index: [chart_index, ...]} in chart order
        unplaced:    [chart_index, ...] charts with zero overlap anywhere,
                     which fall back to the "Charts & Visualisations" appendix
    """
    section_kw = [_section_keywords(sec) for sec in sections]
 
    assignments = {}
    unplaced = []
 
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
 
 
# ── Docx element builders ─────────────────────────────────────────────────────
 
def _add_heading(doc, text, size, color):
    para = doc.add_paragraph()
    run = para.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = color
    para.space_after = Pt(8)
 
 
def _add_body_content(doc, content):
    for line in content.splitlines():
        l = line.strip()
        if not l:
            continue
        # skip any chart placeholder lines that slipped through
        if _CHART_LINE.match(l):
            continue
        if re.match(r'^[-+•]\s+', l):
            para = doc.add_paragraph(style='List Bullet')
            run = para.add_run(re.sub(r'^[-+•]\s+', '', l))
        else:
            para = doc.add_paragraph()
            run = para.add_run(l)
        run.font.size = Pt(11)
        run.font.color.rgb = BODY_COLOR
        para.space_after = Pt(4)
 
 
def _add_table(doc, table_data):
    if not table_data or len(table_data) < 2:
        return
    cols = max(len(r) for r in table_data)
    t = doc.add_table(rows=len(table_data), cols=cols)
    t.style = 'Table Grid'
 
    for ri, row in enumerate(table_data):
        while len(row) < cols:
            row.append("")
        for ci, cell_text in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = cell_text
            run = cell.paragraphs[0].runs[0] if cell.paragraphs[0].runs else cell.paragraphs[0].add_run(cell_text)
            run.font.size = Pt(10)
            if ri == 0:
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                _set_cell_bg(cell, "00205b")
            elif ri % 2 == 0:
                _set_cell_bg(cell, "f2f4f8")
 
    doc.add_paragraph()
 
 
def _set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement('w:shd')
    shd.set(qn('w:val'),   'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'),  hex_color)
    tcPr.append(shd)
 
 
def _add_image(doc, img_path, width_inches=5.5):
    try:
        doc.add_picture(img_path, width=Inches(width_inches))
        last = doc.paragraphs[-1]
        last.alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph()
    except Exception as e:
        print(f"Could not embed image {img_path}: {e}")
 
 
def _add_footer(doc):
    section = doc.sections[0]
    footer  = section.footer
    para    = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
 
    for run in para.runs:
        run.clear()
 
    run_left = para.add_run("HPGPT")
    run_left.font.size  = Pt(9)
    run_left.font.bold  = True
    run_left.font.color.rgb = FOOTER_COLOR
 
    para.add_run("\t")
 
    run_page = para.add_run()
    for tag, text in [('w:fldChar', None), ('w:instrText', 'PAGE'), ('w:fldChar', None)]:
        el = OxmlElement(tag)
        if tag == 'w:fldChar':
            el.set(qn('w:fldCharType'), 'begin' if not run_page._r.findall('.//{%s}fldChar' % 'http://schemas.openxmlformats.org/wordprocessingml/2006/main') else 'end')
        if text:
            el.set(qn('xml:space'), 'preserve')
            el.text = text
        run_page._r.append(el)
 
    run_page.font.size  = Pt(9)
    run_page.font.bold  = True
    run_page.font.color.rgb = FOOTER_COLOR
 
    para.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_ALIGN_PARAGRAPH.RIGHT)