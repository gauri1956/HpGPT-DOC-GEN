import os
import uuid
import logging
import re
import json
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
from mcp.server.fastmcp import FastMCP
 
logging.basicConfig(level=logging.INFO)
 
mcp = FastMCP("hpgpt-document-agent")
 
HP_BLUE  = "#00205b"
HP_RED   = "#e4002b"
CHART_DIR = "outputs/charts"
os.makedirs(CHART_DIR, exist_ok=True)
 
 
# ─── Casing fix-up for auto-generated titles/labels ────────────────────────
 
# Words that must always render fully uppercase (industry/company acronyms).
_ACRONYMS = {
    "hpcl", "bpcl", "iocl", "kl", "lpg", "atf", "inr", "omc", "omcs",
    "ms", "hsd", "sko", "gst", "ytd", "qoq", "yoy", "kpi", "roi", "kpis",
}
 
# Small connector words that should stay lowercase (except as first word).
_LOWERCASE_WORDS = {"and", "or", "of", "in", "vs", "the", "a", "an", "for", "to"}
 
 
def _smart_case(text: str) -> str:
    """
    Fix up auto-generated titles/axis labels so acronyms render uppercase
    (e.g. 'Hpcl Kl' -> 'HPCL KL') instead of naive .title() casing, while
    proper nouns keep normal title case (e.g. 'Reliance Kl' -> 'Reliance KL').
    """
    if not text:
        return text
 
    words = text.split()
    out = []
    for i, word in enumerate(words):
        # Separate trailing punctuation (e.g. "KL," or "Lakh)") from the core word
        match = re.match(r'^(\W*)(\w+)(\W*)$', word)
        if not match:
            out.append(word)
            continue
 
        prefix, core, suffix = match.groups()
        lower_core = core.lower()
 
        if lower_core in _ACRONYMS:
            new_core = core.upper()
        elif i > 0 and lower_core in _LOWERCASE_WORDS:
            new_core = lower_core
        else:
            new_core = core[:1].upper() + core[1:].lower()
 
        out.append(prefix + new_core + suffix)
 
    return " ".join(out)
 
 
def _fmt_value_label(v: float) -> str:
    """
    Format a numeric value for display on a chart bar/point:
    whole numbers show with no decimal point, others show 1 decimal place.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
 
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:,.1f}"
 
 
def _save_fig(fig, name_hint="chart"):
    """Save figure to disk and return path."""
    filename = f"{name_hint}_{uuid.uuid4().hex[:6]}.png"
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logging.info(f"Chart saved: {path}")
    return path
 
 
def _hp_style(ax, title=""):
    """Apply HP brand styling to a matplotlib axes."""
    ax.set_facecolor("#f8f9fc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444", labelsize=9)
    if title:
        ax.set_title(_smart_case(title), color=HP_BLUE, fontsize=12, fontweight="bold", pad=10)
 
 
# ─── Tool 1: generate_chart ─────────────────────────────────────────────────
 
@mcp.tool()
def generate_chart(
    chart_type: str,
    labels: list[str],
    values: list[float],
    title: str = "",
    x_label: str = "",
    y_label: str = ""
) -> dict:
    """
    Generate an HP-branded chart and return the image path.
 
    Args:
        chart_type: One of 'bar', 'line', 'pie', 'horizontal_bar'
        labels:     Category names / x-axis labels
        values:     Numeric data points
        title:      Chart title
        x_label:    X-axis label
        y_label:    Y-axis label
 
    Returns:
        {"path": "<image_path>", "title": "<title>"}
    """
    if len(labels) != len(values):
        return {"error": "labels and values must have the same length"}
 
    # Fix up casing so acronyms like KL/HPCL/IOCL/LPG/ATF/INR render correctly
    title   = _smart_case(title)
    x_label = _smart_case(x_label)
    y_label = _smart_case(y_label)
 
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [HP_BLUE, HP_RED, "#0096d6", "#5c068c", "#007a53",
              "#f5821e", "#1a1a1a", "#6d6e71"]
 
    ct = chart_type.lower()
    value_labels = [_fmt_value_label(v) for v in values]
 
    if ct == "bar":
        bars = ax.bar(labels, values, color=colors[:len(labels)], width=0.6)
        ax.bar_label(bars, labels=value_labels, padding=3, fontsize=9, color="#333")
        if x_label: ax.set_xlabel(x_label, fontsize=10)
        if y_label: ax.set_ylabel(y_label, fontsize=10)
 
    elif ct == "horizontal_bar":
        bars = ax.barh(labels, values, color=colors[:len(labels)], height=0.6)
        ax.bar_label(bars, labels=value_labels, padding=3, fontsize=9, color="#333")
        if x_label: ax.set_xlabel(x_label, fontsize=10)
        if y_label: ax.set_ylabel(y_label, fontsize=10)
 
    elif ct == "line":
        ax.plot(labels, values, color=HP_BLUE, linewidth=2.5,
                marker="o", markersize=6, markerfacecolor=HP_RED)
        ax.fill_between(range(len(labels)), values, alpha=0.08, color=HP_BLUE)
        if x_label: ax.set_xlabel(x_label, fontsize=10)
        if y_label: ax.set_ylabel(y_label, fontsize=10)
 
    elif ct == "pie":
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct="%1.1f%%",
            colors=colors[:len(labels)], startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        for t in autotexts:
            t.set_fontsize(9)
    else:
        return {"error": f"Unknown chart_type '{chart_type}'. Use bar, line, pie, horizontal_bar."}
 
    _hp_style(ax, title)
    fig.tight_layout()
    path = _save_fig(fig, name_hint=ct)
    return {"path": path, "title": title}
 
 
# ─── Tool 2: generate_table_image ──────────────────────────────────────────
 
@mcp.tool()
def generate_table_image(
    headers: list[str],
    rows: list[list[str]],
    title: str = ""
) -> dict:
    """
    Render a data table as an HP-branded image (for embedding in PDFs/PPTs).
 
    Args:
        headers: Column header names
        rows:    List of row data (each row is a list of strings)
        title:   Optional title above the table
 
    Returns:
        {"path": "<image_path>"}
    """
    if not headers or not rows:
        return {"error": "headers and rows cannot be empty"}
 
    headers = [_smart_case(h) for h in headers]
    title = _smart_case(title)
 
    n_rows = len(rows) + 1  # +1 for header
    n_cols = len(headers)
    fig_h  = max(2.5, n_rows * 0.4 + 0.6)
 
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.axis("off")
 
    all_data    = [headers] + rows
    table       = ax.table(cellText=all_data, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(n_cols)))
 
    # HP header styling
    for col in range(n_cols):
        cell = table[0, col]
        cell.set_facecolor(HP_BLUE)
        cell.set_text_props(color="white", fontweight="bold")
 
    # Alternating row colors
    for row in range(1, n_rows):
        for col in range(n_cols):
            table[row, col].set_facecolor("#f2f4f8" if row % 2 == 0 else "white")
            table[row, col].set_text_props(color="#1a1a1a")
 
    if title:
        fig.suptitle(title, color=HP_BLUE, fontsize=11, fontweight="bold", y=0.98)
 
    fig.tight_layout()
    path = _save_fig(fig, name_hint="table")
    return {"path": path}
 
 
# ─── Tool 3: parse_chart_markers ────────────────────────────────────────────
 
@mcp.tool()
def parse_chart_markers(text: str) -> dict:
    """
    Scan LLM-generated markdown for [CHART:...] markers and extract them.
 
    Marker format in LLM output:
        [CHART: bar | title=Revenue | labels=Q1,Q2,Q3 | values=10,20,15]
 
    Returns:
        {"markers": [{"type","title","labels","values","position"}, ...],
         "clean_text": "<text with markers removed>"}
    """
    pattern = re.compile(
        r'\[CHART:\s*(\w+)\s*\|([^\]]+)\]',
        re.IGNORECASE
    )
    markers = []
 
    for match in pattern.finditer(text):
        chart_type = match.group(1).strip()
        params_raw = match.group(2)
        params = {}
        for part in params_raw.split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip().lower()] = v.strip()
 
        labels = [l.strip() for l in params.get("labels", "").split(",") if l.strip()]
        values_raw = params.get("values", "")
        try:
            values = [float(v.strip()) for v in values_raw.split(",") if v.strip()]
        except ValueError:
            values = []
 
        markers.append({
            "type":     chart_type,
            "title":    params.get("title", ""),
            "labels":   labels,
            "values":   values,
            "x_label":  params.get("x_label", ""),
            "y_label":  params.get("y_label", ""),
            "position": match.start()
        })
 
    clean_text = pattern.sub("", text).strip()
    return {"markers": markers, "clean_text": clean_text}
 
 
# ─── Tool 4: summarize_content ──────────────────────────────────────────────
 
@mcp.tool()
def summarize_content(text: str, max_chars: int = 8000) -> dict:
    """
    Trim oversized content intelligently — keeps headings + first sentences
    of each section rather than hard-cutting at a character limit.
 
    Args:
        text:      Input markdown text
        max_chars: Target character budget
 
    Returns:
        {"text": "<summarized text>", "was_trimmed": bool}
    """
    if len(text) <= max_chars:
        return {"text": text, "was_trimmed": False}
 
    lines   = text.splitlines()
    output  = []
    budget  = max_chars
 
    for line in lines:
        if budget <= 0:
            break
        # Always keep headings
        if re.match(r'^#{1,3}\s+', line):
            output.append(line)
            budget -= len(line)
        elif len(line) <= budget:
            output.append(line)
            budget -= len(line)
        else:
            # Take what fits, end at sentence boundary
            chunk = line[:budget]
            last_period = chunk.rfind(". ")
            if last_period > 50:
                chunk = chunk[:last_period + 1]
            output.append(chunk + " [...]")
            budget = 0
 
    return {"text": "\n".join(output), "was_trimmed": True}
 
 
# ─── Entry point ────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    mcp.run()
 