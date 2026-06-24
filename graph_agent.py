import re
import logging
from mcp_server import generate_chart, generate_table_image, parse_chart_markers
 
logger = logging.getLogger(__name__)
 
# Human-readable label map for known column names.
# Keys are normalised to lowercase, underscore-separated form
# (see _normalise_key) so they match regardless of whether the
# incoming column name uses spaces or underscores.
COL_LABEL_MAP = {
    'petrol_kl':        'Petrol Sales',
    'diesel_kl':        'Diesel Sales',
    'lpg_mt':           'LPG Sales',
    'lubricants_kl':    'Lubricants Sales',
    'revenue_inr_lakh': 'Revenue (INR Lakh)',
    'opening_stock':    'Product-wise Opening Stock',
    'received':         'Product-wise Received Quantity',
    'issued':           'Product-wise Issued Quantity',
    'closing_stock':    'Product-wise Closing Stock',
    'reorder_level':    'Reorder Levels',
    'hpcl_kl':          'HPCL Sales',
    'bpcl_kl':          'BPCL Sales',
    'iocl_kl':          'IOCL Sales',
    'reliance_kl':      'Reliance Sales',
    'nayara_kl':        'Nayara Sales',
}
 
# Known abbreviations/units we always want fully uppercased in chart
# titles & axis labels, no matter what casing _clean_col / .title()
# produced for them. This keeps output looking professional for
# columns that AREN'T in COL_LABEL_MAP (i.e. any arbitrary file).
_ABBREV_FIX = {
    'kl', 'mt', 'inr', 'lakh', 'usd', 'pct', 'gst', 'atf', 'lpg',
    'omc', 'hp', 'hpcl', 'bpcl', 'iocl', 'cr', 'crore', 'gdp',
    'roi', 'ytd', 'mom', 'yoy', 'sku', 'id',
}
 
# Words ignored when matching titles for similarity checks (not currently
# used for dedup, kept here in case future title-based grouping is added).
_TITLE_STOPWORDS = {
    'by', 'of', 'the', 'a', 'an', 'comparison', 'trend', 'trends',
    'analysis', 'product', 'monthly', 'data', 'chart', 'overview',
    'over', 'time', 'across', 'vs', 'and',
}
 
# Unit-like tokens recognised when building "comparable group" chart
# titles (see _comparable_groups_chart).
_UNIT_TOKENS = {
    'kl', 'mt', 'inr', 'lakh', 'usd', 'pct', 'percent', '%',
    'rs', 'cr', 'crore', 'tonnes', 'tons', 'units',
}
 
 
def _normalise_key(col_name):
    """lowercase + underscore-separated, regardless of input spacing."""
    return re.sub(r'\s+', '_', col_name.strip().lower())
 
 
def _fix_abbrev_casing(text):
    """Uppercase known abbreviation/unit words inside a label."""
    words = text.split()
    return ' '.join(w.upper() if w.lower() in _ABBREV_FIX else w for w in words)
 
 
def _readable_label(col_name):
    key = _normalise_key(col_name)
    if key in COL_LABEL_MAP:
        return COL_LABEL_MAP[key]
 
    # Strip a trailing unit token, whether it's underscore- or
    # space-separated from the rest of the name (e.g. "Sales_USD"
    # or "Sales USD" -> "Sales").
    label = re.sub(r'[\s_](kl|mt|inr|lakh|usd|pct|%)$', '', col_name, flags=re.IGNORECASE)
    label = label.replace('_', ' ').strip().title()
    return _fix_abbrev_casing(label)
 
 
def _smart_chart_type(label_col, value_col, n_labels):
    lc = label_col.lower()
    vc = value_col.lower()
    # Time-series → line
    if any(x in lc for x in ('month', 'date', 'year', 'period', 'quarter', 'week', 'day')):
        return 'line'
    # Too many categories → horizontal bar
    if n_labels > 8:
        return 'horizontal_bar'
    # Share/percent → pie
    if any(x in vc for x in ('share', 'pct', 'percent', 'ratio', 'proportion')):
        return 'pie'
    return 'bar'
 
 
def generate_charts_from_data(data, instruction=""):
    """
    Generate charts from extracted file data.
    Works for any file type that produces chart_candidates.
    For multi-sheet Excel, iterates all sheets.
 
    Returns a list of (title, path, value_signature) tuples.
    `value_signature` is used purely for cross-pipeline dedup —
    callers that don't need it can ignore the third element.
    """
    chart_images = []
 
    # Handle multi-sheet Excel
    if data.get('type') == 'multi_sheet':
        for sheet_name, sheet_data in data.get('sheets', {}).items():
            sheet_charts = _charts_from_candidates(
                sheet_data.get('chart_candidates', []),
                data.get('file', 'file')
            )
            chart_images.extend(sheet_charts)
        return chart_images
 
    candidates = data.get('chart_candidates', [])
    file_name = data.get('file', 'file')
    chart_images = _charts_from_candidates(candidates, file_name)
    return chart_images
 
 
def _generate_grouped_line_chart(group, file_name):
    # Find the labels (from the first column in the group)
    first_col = group['columns'][0]
    labels = [str(l) for l in first_col['labels']]
    
    # Build values dict: {entity_name: values}
    values_dict = {}
    for entity, c in zip(group['entities'], group['columns']):
        values_dict[entity] = [float(v) for v in c['values']]
        
    suffix_clean = _fix_abbrev_casing(group['suffix'])
    title = f"Monthly Sales Comparison ({suffix_clean})" if "sales" in group['suffix'].lower() else f"Monthly comparison of {suffix_clean}"
    if file_name and "omc" in file_name.lower():
        title = "Monthly Sales Comparison (HPCL vs BPCL vs IOCL vs Reliance vs Nayara)"
        
    y_label = _fix_abbrev_casing(group['suffix'].title())
    
    try:
        result = generate_chart(
            chart_type='line',
            labels=labels,
            values=values_dict,  # Pass the dict here!
            title=title,
            y_label=y_label
        )
        if 'path' in result:
            sig = tuple(round(sum(vals), 2) for vals in values_dict.values())
            return (title, result['path'], sig)
    except Exception as e:
        logger.warning(f"Grouped line chart failed: {e}")
    return None
 
 
def _charts_from_candidates(candidates, file_name):
    chart_images = []
    if not candidates:
        logger.info(f"No chart candidates in {file_name}")
        return chart_images
 
    # First, find comparable groups
    groups = _find_comparable_groups(candidates)
    grouped_cols = set()
    
    # Generate grouped line charts for time-series siblings
    for group in groups:
        label_col = group['label_col'].lower()
        is_time_series = any(x in label_col for x in ('month', 'date', 'year', 'period', 'quarter', 'week', 'day'))
        
        if is_time_series:
            grouped_chart = _generate_grouped_line_chart(group, file_name)
            if grouped_chart:
                chart_images.append(grouped_chart)
                # Mark these columns as grouped so we don't generate individual charts for them
                for c in group['columns']:
                    grouped_cols.add(c['value_col'])
 
    for c in candidates[:8]:   # up to 8 individual charts
        vcol = c['value_col']
        if vcol in grouped_cols:
            continue
 
        labels = [str(l) for l in c['labels']]
        values = [float(v) for v in c['values']]
        label_col = c['label_col']
 
        title = _readable_label(vcol)
        chart_type = _smart_chart_type(label_col, vcol, len(labels))
        y_label = _fix_abbrev_casing(vcol.replace('_', ' ').title())
 
        try:
            result = generate_chart(
                chart_type=chart_type,
                labels=labels,
                values=values,
                title=title,
                y_label=y_label
            )
            if 'path' in result:
                sig = tuple(round(v, 2) for v in values)
                chart_images.append((title, result['path'], sig))
                logger.info(f"Chart [{chart_type}]: {title} -> {result['path']}")
        except Exception as e:
            logger.warning(f"Chart generation failed for {title}: {e}")
 
    # One extra chart comparing averages (bar chart)
    # Only run it for groups that weren't already plotted as time-series line charts
    non_time_series_candidates = [c for c in candidates if c['value_col'] not in grouped_cols]
    if non_time_series_candidates:
        chart_images.extend(_comparable_groups_chart(non_time_series_candidates, file_name))
 
    return chart_images
 
 
def _find_comparable_groups(candidates, min_group=3):
    """
    Detect groups of 3+ sibling numeric columns (sharing the same x-axis /
    label_col) that represent the SAME metric measured for DIFFERENT
    entities — e.g. columns named "HPCL KL", "BPCL KL", "IOCL KL"
    (shared trailing word "KL" = the metric/unit, distinct leading words
    = the entity names), or "North Region Revenue", "South Region
    Revenue", "East Region Revenue" (shared trailing "Region Revenue").
 
    This is purely name-pattern based, so it works for ANY dataset —
    and it deliberately does NOT fire for groups of differently-named
    metrics on the same entity (e.g. "Opening Stock", "Received",
    "Issued"), since those share no common trailing word.
    """
    by_label = {}
    for c in candidates:
        by_label.setdefault(c['label_col'], []).append(c)
 
    groups = []
    for label_col, cols in by_label.items():
        if len(cols) < min_group:
            continue
 
        token_lists = [c['value_col'].split() for c in cols]
        min_len = min(len(t) for t in token_lists)
 
        # Find the longest common trailing-word sequence, leaving at
        # least one leading word per column (so there's still an
        # "entity name" left over).
        suffix_len = 0
        for i in range(1, min_len):
            if len({tuple(t[-i:]) for t in token_lists}) == 1:
                suffix_len = i
            else:
                break
 
        if suffix_len == 0:
            continue
 
        prefixes = [' '.join(t[:-suffix_len]) for t in token_lists]
        if any(not p for p in prefixes) or len(set(prefixes)) < min_group:
            continue
 
        suffix = ' '.join(token_lists[0][-suffix_len:])
        groups.append({
            'label_col': label_col,
            'suffix': suffix,
            'entities': prefixes,
            'columns': cols,
        })
 
    return groups
 
 
def _comparable_groups_chart(candidates, file_name):
    """Build one bar chart per comparable-entity group (see above)."""
    extra = []
 
    for group in _find_comparable_groups(candidates):
        labels, values = [], []
        for entity, c in zip(group['entities'], group['columns']):
            vals = [float(v) for v in c['values']]
            if not vals:
                continue
            labels.append(_fix_abbrev_casing(entity))
            values.append(round(sum(vals) / len(vals), 2))
 
        if len(labels) < 3:
            continue
 
        suffix_words = group['suffix'].split()
        if suffix_words[-1].lower() in _UNIT_TOKENS:
            unit = suffix_words[-1].upper()
            metric = ' '.join(suffix_words[:-1]).strip()
            if metric:
                title = f"Average {metric} Comparison ({unit})"
            else:
                title = f"Average Values Comparison ({unit})"
            y_label = f"Average ({unit})"
        else:
            suffix_clean = _fix_abbrev_casing(group['suffix'])
            title = f"Average {suffix_clean} — Comparison"
            y_label = f"Average {suffix_clean}"
 
        try:
            result = generate_chart(
                chart_type='bar',
                labels=labels,
                values=values,
                title=title,
                y_label=y_label
            )
            if 'path' in result:
                sig = tuple(values)
                extra.append((title, result['path'], sig))
                logger.info(f"Comparison chart [{file_name}]: {title} -> {result['path']}")
        except Exception as e:
            logger.warning(f"Comparison chart generation failed: {e}")
 
    return extra
 
 
def generate_charts_from_markers(text):
    """
    Parse [CHART:...] markers from LLM output and generate images.
 
    Returns (clean_text, chart_images) where chart_images is a list of
    (title, path, value_signature) tuples — same shape as
    generate_charts_from_data, so the two can be merged and deduped
    together.
    """
    result = parse_chart_markers(text)
    markers = result.get('markers', [])
    clean_text = result.get('clean_text', text)
    chart_images = []
 
    for m in markers:
        if not m['labels'] or not m['values']:
            continue
        if len(m['labels']) != len(m['values']):
            continue
        try:
            r = generate_chart(
                chart_type=m['type'],
                labels=m['labels'],
                values=m['values'],
                title=m.get('title', ''),
                x_label=m.get('x_label', ''),
                y_label=m.get('y_label', '')
            )
            if 'path' in r:
                sig = tuple(round(float(v), 2) for v in m['values'])
                chart_images.append((m.get('title', ''), r['path'], sig))
                logger.info(f"Marker chart: {m.get('title')} -> {r['path']}")
        except Exception as e:
            logger.warning(f"Marker chart failed: {e}")
 
    return clean_text, chart_images
 
 
def dedup_charts(chart_list, seen_signatures=None, tolerance=0.01):
    """
    Drop charts whose underlying numeric data has already been charted
    elsewhere in the document.
 
    `seen_signatures` is a set of value-signature tuples accumulated
    across the whole run — pass it in and use the returned (updated)
    set for the next call, so duplicates are caught across files and
    sections, not just within one chart_list.
 
    Two signatures are treated as the same chart if they have the same
    length and every value matches within `tolerance` (handles tiny
    rounding differences between the deterministic pipeline and the
    LLM's marker output).
 
    Charts with an empty/missing signature are always kept (nothing
    to compare).
 
    Returns (deduped_chart_list, updated_seen_signatures).
    """
    if seen_signatures is None:
        seen_signatures = set()
 
    deduped = []
    for title, path, sig in chart_list:
        if not sig:
            deduped.append((title, path, sig))
            continue
 
        is_dup = sig in seen_signatures
        if not is_dup:
            for existing in seen_signatures:
                if len(existing) == len(sig) and all(
                    abs(a - b) <= tolerance for a, b in zip(existing, sig)
                ):
                    is_dup = True
                    break
 
        if is_dup:
            logger.info(f"Dropping duplicate chart: {title}")
            continue
 
        seen_signatures.add(sig)
        deduped.append((title, path, sig))
 
    return deduped, seen_signatures
 