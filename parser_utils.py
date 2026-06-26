import re

def clean_text(text):
    """Clean markdown styling helper"""
    if not text:
        return ""
    # Strip asterisks and underscores
    return re.sub(r'[*_]+', '', text).strip()

def parse_sections(text):
    """
    Parse sections from markdown body.
    Returns a list of dict: [{'level': int, 'heading': str, 'body': str, 'table': list[list[str]] or None}]
    """
    if not text:
        return []
    pat = re.compile(r'^(#{1,3})\s+(.*)$', re.MULTILINE)
    ms  = list(pat.finditer(text))
    if not ms:
        return [{'level': 0, 'heading': '', 'body': text.strip(), 'table': None}]
    secs = []
    if ms[0].start() > 0:
        pre = text[:ms[0].start()].strip()
        if pre:
            secs.append({'level': 0, 'heading': '', 'body': pre, 'table': None})
    for i, m in enumerate(ms):
        body = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)].strip()
        secs.append({
            'level':   len(m.group(1)),
            'heading': m.group(2).strip(),
            'body':    body,
            'table':   extract_table(body),
        })
    return secs

def extract_table(text):
    """
    Extract first markdown table found in the text body as a list of rows.
    Supports empty cells and preserves column alignment.
    """
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if '|' in l]
    if len(lines) < 2:
        return None
    # Filter out table header delimiter line (e.g. |---|---|)
    rows = [l for l in lines if not re.match(r'^[\|\s\-:]+$', l)]
    if len(rows) < 2:
        return None
    
    # Process cells
    table_data = []
    for r in rows:
        cells = [c.strip() for c in r.split('|')]
        # Strip leading and trailing empty cell from the splitting of border pipes
        if r.startswith('|') and len(cells) > 0 and cells[0] == '':
            cells = cells[1:]
        if r.endswith('|') and len(cells) > 0 and cells[-1] == '':
            cells = cells[:-1]
        
        table_data.append(cells)
        
    return table_data

def strip_table_from_text(text):
    """
    Returns text with all markdown table lines removed.
    """
    if not text:
        return ""
    lines = [l for l in text.splitlines() if '|' not in l]
    return "\n".join(lines).strip()
