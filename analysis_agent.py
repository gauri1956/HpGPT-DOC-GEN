import json
import logging
import re
from collections import Counter
 
from llm_agent import query_llama, get_doctype_rules, get_cross_file_rules, PROMPT_TOKEN_BUDGET, get_system_prompt
from data_fusion_agent import render_context
 
logger = logging.getLogger(__name__)
 
 
# -- Intent-specific DIGEST instructions --------------------------------------
_INTENT_INSTRUCTIONS = {
    "analytical": (
        "INTENT: This is a NUMERICAL / KPI dataset. Focus on:\n"
        "- Extracting specific metrics, rates, totals, averages with exact numbers.\n"
        "- Noting trends, gaps, or outliers visible in the data.\n"
        "- Flagging which numbers are above/below average or notable.\n"
        "- Identifying the root cause behind each notable metric (one sentence each).\n"
        "- Noting which findings would cross-correlate with HR, operational, or\n"
        "  financial data from other files in the same report.\n"
        "- Chart markers are highly encouraged where numeric data supports them.\n"
    ),
    "informational": (
        "INTENT: This is an INFORMATIONAL / REFERENCE file. Focus on:\n"
        "- Summarising the key facts and topics covered in the file.\n"
        "- Do NOT invent analysis, recommendations, or KPIs not present in the data.\n"
        "- Do NOT write 'Root Cause', 'Risk', or 'Strategic Implication' bullets.\n"
        "- Present facts as-is: what the document says, what categories exist,\n"
        "  what the key points are.\n"
        "- Chart markers only if explicit numeric data is present -- otherwise skip.\n"
    ),
    "educational": (
        "INTENT: This is EDUCATIONAL / TRAINING content. Focus on:\n"
        "- Identifying the learning topics, modules, or concepts covered.\n"
        "- Noting any scores, completion rates, or assessment data if present.\n"
        "- Do NOT reframe educational content as a business KPI report.\n"
        "- Do NOT write Executive Summary, Risk Factors, or Recommendations\n"
        "  unless the user explicitly asked for them.\n"
        "- Summarise what is being taught / learned, not business performance.\n"
        "- Note the intended audience and learning outcomes where evident.\n"
        "- Chart markers only if assessment scores or numeric data is present.\n"
    ),
    "policy": (
        "INTENT: This is a POLICY / PROCEDURE / GUIDELINE document. Focus on:\n"
        "- Listing the key rules, requirements, or steps described.\n"
        "- Noting scope, applicability, and compliance requirements.\n"
        "- Identifying who is responsible for each obligation.\n"
        "- Do NOT reframe as a performance analysis or KPI report.\n"
        "- Do NOT add recommendations unless they are explicitly in the source.\n"
        "- Present the policy content accurately and completely.\n"
        "- Chart markers only if the policy contains numeric thresholds or data.\n"
    ),
    "operational": (
        "INTENT: This is OPERATIONAL / TRANSACTIONAL data (stock, inventory, logs). Focus on:\n"
        "- Reporting current status: opening/closing balances, quantities, totals.\n"
        "- Flagging anomalies: missing entries, zero values, large variances.\n"
        "- Computing simple aggregates: totals, averages, min/max per category.\n"
        "- Do NOT write strategic recommendations or HR-style analysis.\n"
        "- Chart markers are encouraged for quantity comparisons across categories.\n"
    ),
}
 
# -- Base digest prompt -------------------------------------------------------
# KEY CHANGE: Each bullet now requires a [src: ...] tag so the orchestrator's
# _validate_digest() can check that every numeric claim has a traceable source.
# This is the structured output enforcement — no extra LLM call, just a format
# constraint added to the existing digest prompt.
_BASE_DIGEST_PROMPT = """
You are a data digest agent. You receive structured data extracted from ONE file and
must extract FACTS ONLY for use by another agent that will write the final report.
 
STRICT RULES:
- Output 4-8 bullet points (- ), each ONE complete sentence stating a specific fact
  with an EXACT number taken directly from the data provided, or a simple sum/average/
  difference/ratio of numbers given in the data.
- Use clean readable names -- never raw column names like HPCLKL, OpeningStock,
  PetrolKL. Write instead: HPCL Sales (KL), Opening Stock, Petrol Sales (KL).
- NEVER estimate, infer, or invent a metric unless it is literally computable from
  numbers given in the data via a stated calculation.
- Round numbers to at most 1 decimal place.
 
SOURCE TAGGING -- MANDATORY:
- Every bullet that contains a number MUST end with a source tag in this exact format:
  [src: <column or sheet name>]
  Examples:
    - Total Petrol Sales (KL) across all months is 4,820 KL. [src: Petrol KL]
    - Average Salary in Finance department is Rs. 72,000. [src: Salary, Department]
    - Opening Stock of Diesel in April is 1,200 MT. [src: Opening Stock, Sheet: April]
- If a number comes from a calculation across columns, list all source columns:
  [src: Opening Stock + Received - Issued]
- For text/policy files with no numbers, the [src:] tag is not required.
- CONFIDENCE: If a metric is derived (not directly in a cell) add [conf: derived].
  If directly read from a cell, add [conf: direct]. Example:
  - Attrition rate is 12.3% based on resignations vs headcount. [src: Attrition, Headcount] [conf: derived]
 
CHART RULES:
- Where a chart adds value, insert it on its own line:
  [CHART: bar | title=Salary by Employee | labels=Rahul,Priya,Amit | values=65000,55000,72000 | y_label=Salary]
  Only chart numeric data actually present in the data.
- Do NOT generate charts for ID columns, serial numbers, or index columns.
 
OTHER RULES:
- Do NOT mention internal terms like "chart candidates", "numeric columns", filenames, or UUIDs.
- IMPORTANT: The "Instruction" below may describe a full multi-section report. IGNORE
  that structure -- it applies to the FINAL report. This step ONLY extracts facts.
- Do NOT write any heading, title, Executive Summary, KPI Dashboard, Recommendations,
  or narrative framing. Output ONLY bullet points (and chart markers).
- Maximum 250 words.
"""
 
# -- Intent-specific INSIGHT (final report) instructions ----------------------
_INSIGHT_INTENT_INSTRUCTIONS = {
    "analytical": (
        "DOCUMENT INTENT -- ANALYTICAL REPORT:\n"
        "Write a full analytical report with Executive Summary, Key Findings,\n"
        "root-cause analysis, cross-dataset correlation, Risk Assessment, and\n"
        "Recommendations. Apply all grounding and strategic thinking rules from\n"
        "your system prompt.\n"
    ),
    "informational": (
        "DOCUMENT INTENT -- INFORMATIONAL DOCUMENT:\n"
        "Write a clear, factual informational document. Structure it with an Introduction,\n"
        "Key Topics / Sections derived from the content, and a Summary.\n"
        "Do NOT add recommendations, risk analysis, or KPI dashboards unless the user\n"
        "explicitly asked for them. Present information accurately -- do not editorialize\n"
        "or invent business consequences not supported by the source material.\n"
    ),
    "educational": (
        "DOCUMENT INTENT -- EDUCATIONAL / TRAINING DOCUMENT:\n"
        "Write a clear educational document or presentation covering the topics, modules,\n"
        "or concepts in the source material. Structure it with an Introduction,\n"
        "topic-by-topic sections, and a Summary / Key Takeaways.\n"
        "Do NOT reframe educational content as a KPI dashboard or HR analytics report.\n"
        "Do NOT add Executive Summary, Attrition Analysis, or business KPI sections\n"
        "unless the user explicitly asked for them. Focus on what is being taught.\n"
    ),
    "policy": (
        "DOCUMENT INTENT -- POLICY / PROCEDURE DOCUMENT:\n"
        "Write a structured policy or procedure document. Include: Purpose & Scope,\n"
        "Key Requirements / Rules, Procedures / Steps, Responsibilities, and Compliance.\n"
        "Do NOT add analytical recommendations or KPI dashboards.\n"
        "Present the policy content faithfully.\n"
    ),
    "operational": (
        "DOCUMENT INTENT -- OPERATIONAL STATUS REPORT:\n"
        "Write an operational status report covering current inventory / transaction /\n"
        "log status. Include: Summary of current status, Category-wise breakdown,\n"
        "Anomalies or items requiring attention, and Operational Recommendations\n"
        "(only those directly supported by the data).\n"
        "Do NOT write strategic HR or financial analysis unless the data supports it.\n"
    ),
    "training_material": (
        "DOCUMENT INTENT -- TRAINING MATERIAL:\n"
        "Write a full training document with Learning Objectives, topic-by-topic\n"
        "concept explanations (not bullet dumps), real-world HPCL examples, a Case Study,\n"
        "Assessment questions with answers, and a Glossary.\n"
        "Do NOT reframe as a KPI report.\n"
    ),
    "sop": (
        "DOCUMENT INTENT -- STANDARD OPERATING PROCEDURE:\n"
        "Write a numbered-step SOP with Purpose, Prerequisites, Procedure steps\n"
        "(Action -> Role -> Tool -> Safety Note), Critical Control Points, Safety Warnings,\n"
        "Troubleshooting, and Performance Metrics.\n"
    ),
    "business_report": (
        "DOCUMENT INTENT -- BUSINESS ANALYTICAL REPORT:\n"
        "Write a full executive-grade report: Executive Summary -> KPI Dashboard\n"
        "-> Key Findings (with root cause and impact) -> Risk Assessment -> Strategic\n"
        "Recommendations (each tied to a specific finding). Every finding must answer\n"
        "What / Why / Why it matters / What risk / What action.\n"
    ),
    "financial_report": (
        "DOCUMENT INTENT -- FINANCIAL REPORT:\n"
        "Write an executive financial report: Executive Summary -> Financial Snapshot\n"
        "-> Variance Analysis -> Segment Analysis -> Risk Assessment -> Forward Outlook\n"
        "-> Priority Actions. Every variance must state the root cause.\n"
    ),
 
    # -- Official PSU/Government document intents --
    "file_note": (
        "DOCUMENT INTENT -- FILE NOTE (Official PSU/Government Document):\n"
        "This is a formal internal noting document used for decision-making in HPCL.\n"
        "Do NOT write an analytical report, executive summary, or KPI dashboard.\n"
        "Follow the official File Note format exactly as specified in the format block above.\n"
        "Use formal government language. Number every paragraph. End with Approval Sought.\n"
        "Every section MUST have substantive numbered paragraphs -- not one-liners.\n"
    ),
    "office_memorandum": (
        "DOCUMENT INTENT -- OFFICE MEMORANDUM / O.M. (Official PSU/Government Document):\n"
        "This is a formal inter-departmental communication document.\n"
        "Do NOT write a report or essay. Follow the official O.M. format exactly.\n"
        "Open with 'The undersigned is directed to...' and use numbered paragraphs.\n"
        "The Body section must have at least 3 substantive numbered paragraphs.\n"
    ),
    "office_notice": (
        "DOCUMENT INTENT -- OFFICE NOTICE (Official PSU/Government Document):\n"
        "This is a formal notice to staff or departments.\n"
        "Do NOT write a report or essay. Follow the official Notice format exactly.\n"
        "Open with 'It is hereby notified that...' and use numbered points.\n"
        "The Notice Body must contain at least 4 numbered directives or information points.\n"
    ),
    "circular": (
        "DOCUMENT INTENT -- OFFICE CIRCULAR (Official PSU/Government Document):\n"
        "This is a formal policy/procedure circular issued to all departments.\n"
        "Do NOT write a report or essay. Follow the official Circular format exactly.\n"
        "Use numbered instructions in the Instructions section -- minimum 5 instructions.\n"
        "State effective date and issuing authority explicitly.\n"
    ),
    "purchase_note": (
        "DOCUMENT INTENT -- PURCHASE / PROCUREMENT NOTE (Official PSU/Government Document):\n"
        "\n"
        "This is a PROCUREMENT JUSTIFICATION document. It must read as a complete,\n"
        "professionally written government procurement noting -- not a template or form.\n"
        "\n"
        "MANDATORY CONTENT REQUIREMENTS:\n"
        "- Section 1 (Purpose and Operational Requirement): Write 3-5 numbered paragraphs\n"
        "  explaining exactly WHY this item/service is needed, which operational gap it fills,\n"
        "  which department/process uses it, what the current situation is without it, and\n"
        "  how it aligns with HPCL's operational objectives. Be specific and substantive.\n"
        "\n"
        "- Section 2 (Specification of Items): Provide a complete markdown table with\n"
        "  columns: S.No | Item/Service Description | Technical Specification | Quantity |\n"
        "  Est. Unit Cost (Rs.) | Total Est. Cost (Rs.).\n"
        "  Use 'Rs. __________________' for unknown costs.\n"
        "\n"
        "- Section 3 (Operational Justification and Risk of Non-Procurement): Write 3-5\n"
        "  numbered paragraphs explaining WHY these specific specifications are required,\n"
        "  what alternatives were considered and rejected, and what SPECIFIC operational,\n"
        "  safety, compliance, or financial risks arise if this procurement is NOT approved.\n"
        "\n"
        "- Section 4 (Budget Provision): Write 2-3 numbered paragraphs on the budget head,\n"
        "  sanctioned budget, and estimated expenditure. Use 'Rs. __________________' for unknowns.\n"
        "\n"
        "- Section 5 (Procurement Method): Name and justify the recommended procurement\n"
        "  method (Open Tender / Limited Tender / Single Source / GeM / Rate Contract).\n"
        "  Reference GFR 2017 or HPCL procurement guidelines.\n"
        "\n"
        "- Section 6 (Delivery Schedule): State expected timeline and milestones.\n"
        "\n"
        "- Section 7 (Recommendation and Approval Sought): Write a formal recommendation\n"
        "  paragraph and state EXACTLY what sanction/approval is being sought.\n"
        "  End with the full approval chain: Prepared by / Recommended by / Approved by.\n"
        "\n"
        "STYLE RULES:\n"
        "- Every section must use numbered paragraphs (1., 2., 3.) -- NO bullet points.\n"
        "- Use formal government prose throughout.\n"
        "- Use 'the undersigned', 'it is submitted', 'competent authority' etc.\n"
        "- Unknown values: use '__________________' underlines, never brackets.\n"
        "- Do NOT invent specific costs, quantities, person names, or reference numbers.\n"
    ),
    "office_order": (
        "DOCUMENT INTENT -- OFFICE ORDER (Official PSU/Government Document):\n"
        "This is a formal order issued by competent authority.\n"
        "Do NOT write a report or essay. Follow the official Office Order format exactly.\n"
        "Open with 'It is ordered that...' and number all paragraphs.\n"
        "The Order section must have at least 3 numbered paragraphs.\n"
    ),
}
 
 
def _build_digest_prompt(intent: str) -> str:
    """Combine base digest prompt with intent-specific instructions."""
    intent_block = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["informational"])
    return _BASE_DIGEST_PROMPT.strip() + "\n\n" + intent_block.strip()
 
 
# ==============================================================================
# Strip [src:] and [conf:] tags from digest text.
# Called in TWO places:
#   1. run_analysis() return — so the CACHED digest (used for chart extraction
#      and passed to run_insight) is clean. Orchestrator calls _validate_digest()
#      on the RAW result BEFORE stripping, then strips before caching.
#   2. run_insight() — belt-and-suspenders strip on whatever comes in, in case
#      a cached digest already had tags stripped or still has some.
# ==============================================================================
 
def _strip_digest_tags(digest_text: str) -> str:
    """
    Remove [src: ...] and [conf: ...] tags from digest text.
    Called before digest_results are passed to run_insight() so the LLM
    doesn't see or reproduce the internal validation tags in the final document.
    Also called by run_analysis() so the returned digest is tag-free for caching.
    """
    cleaned = re.sub(r'\[src:[^\]]*\]', '', digest_text)
    cleaned = re.sub(r'\[conf:[^\]]*\]', '', cleaned)
    # Tidy up any double spaces left behind
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()
 
 
def _compress_digest(digest_text: str) -> str:
    """
    Compresses a digest text to retain only critical structured facts.
    - Preserves bullet points (lines starting with '-')
    - Preserves heading indicators (lines starting with '#')
    - Preserves chart references (lines containing '[CHART:')
    - Filters out conversational introductions, verbose paragraphs, and concluding remarks.
    """
    if not digest_text:
        return ""
    
    compressed_lines = []
    for line in digest_text.splitlines():
        s = line.strip()
        if not s:
            continue
        
        # Keep headings, bullet points, and chart references
        if s.startswith('#') or s.startswith('-') or s.startswith('*') or '[CHART:' in s:
            compressed_lines.append(line)
        # Keep short lines containing metric values (numeric digits) if they look like findings
        elif any(c.isdigit() for c in s) and len(s) < 150:
            compressed_lines.append("- " + s if not s.startswith("-") else s)
            
    return "\n".join(compressed_lines).strip()
 
 
# ==============================================================================
# Extract structured facts from digest for validation.
# Called by orchestrator._validate_digest() on the RAW tagged digest
# (before tags are stripped) so [src:] presence can be checked.
# ==============================================================================
 
def _extract_digest_facts(digest_text: str, file_name: str) -> list[dict]:
    """
    Parse digest bullets into structured fact records.
 
    Each record:
        {
            "text":       str,    # full bullet text
            "has_number": bool,   # contains a numeric value
            "has_src":    bool,   # has [src: ...] tag
            "confidence": str,    # "direct" | "derived" | "unknown"
            "src_ref":    str,    # extracted source reference
            "file":       str,    # source file name
            "supported":  bool,   # True if has_number -> has_src
        }
 
    Used by orchestrator._validate_digest() for structured checking.
    """
    facts = []
    lines = [l.strip() for l in digest_text.splitlines() if l.strip().startswith("- ")]
 
    for line in lines:
        has_number = bool(re.search(r'\b\d+[\.,]?\d*\b', line))
 
        src_match = re.search(r'\[src:\s*([^\]]+)\]', line, re.IGNORECASE)
        has_src   = src_match is not None
        src_ref   = src_match.group(1).strip() if src_match else ""
 
        conf_match = re.search(r'\[conf:\s*(\w+)\]', line, re.IGNORECASE)
        confidence = conf_match.group(1).lower() if conf_match else "unknown"
 
        # A claim is supported if it either has no number (text fact)
        # or has a number AND a [src:] tag
        supported = (not has_number) or has_src
 
        facts.append({
            "text":       line,
            "has_number": has_number,
            "has_src":    has_src,
            "confidence": confidence,
            "src_ref":    src_ref,
            "file":       file_name,
            "supported":  supported,
        })
 
    return facts
 
 
def run_analysis(data, instruction, intent: str = None, dataset_role: str = None,
                 causal_hints: list = None, prior_digests_summary: str = None):
    """
    Extract a fact-digest from a single dataset. Output is bullet points with
    real numbers (and optional chart markers) -- no report structure.
 
    KEY DESIGN:
    - The LLM is prompted to include [src: <column>] and [conf: direct|derived]
      tags on every numeric bullet.
    - The RAW tagged result is returned so the orchestrator can call
      _validate_digest() on it BEFORE stripping. The orchestrator is responsible
      for stripping tags before passing to run_insight() and before caching.
    - This function does NOT strip tags — that separation keeps _validate_digest()
      able to do its job on the raw output.
 
    Returns: (tagged_digest_text, intent)
        tagged_digest_text: raw digest WITH [src:]/[conf:] tags intact
        intent:             detected or passed-in intent string
    """
    from data_agent import detect_content_intent
 
    if isinstance(data, dict):
        clean_name = re.sub(r'^[a-f0-9]{8,}_', '', data.get('file', ''))
        clean_name = (clean_name.replace('.xlsx', '').replace('.csv', '')
                                .replace('_', ' ').strip())
        data = {**data, 'file': clean_name}
 
    if intent is None:
        intent, confidence = detect_content_intent(data, user_prompt=instruction)
        logger.info(
            f"Detected intent for {data.get('file', 'unknown')}: "
            f"{intent} (confidence={confidence})"
        )
 
    digest_prompt = _build_digest_prompt(intent)
    
    def estimate_tokens(prompt_str: str) -> int:
        return (len(digest_prompt) + len(prompt_str)) // 4
 
    trend_lines = []
    if isinstance(data, dict):
        trends = data.get("trends", {})
        if trends:
            for col, t in trends.items():
                trend_lines.append(f"  - Column '{col}': {t.get('trend_signal', '')}")
 
    if isinstance(data, dict) and data.get("type") in ("tabular", "multi_sheet"):
        smart_data = {
            "file": data.get("file"),
            "type": data.get("type"),
            "columns": data.get("columns"),
            "stats": data.get("stats"),
            "entity_stats": data.get("entity_stats"),
            "trends": data.get("trends"),
            "sample": data.get("sample"),
        }
        if "sheets_summary" in data:
            smart_data["sheets_summary"] = data["sheets_summary"]
        if "cross_sheet_entities" in data:
            smart_data["cross_sheet_entities"] = data["cross_sheet_entities"]
 
        def build_prompt_with_data(sd):
            ds = json.dumps(sd, indent=2, default=str)
            p_lines = [
                f"Instruction (for context only -- see rules above about ignoring its structure): {instruction}",
                f"Dataset: {data.get('file', 'unknown')}"
            ]
            if dataset_role:
                p_lines.append(f"Dataset Role: {dataset_role}")
            if causal_hints:
                hints_str = "\n".join(f"- {h}" for h in causal_hints)
                p_lines.append(f"Causal Hints / Expected Relationships:\n{hints_str}")
            if trend_lines:
                p_lines.append(
                    "Detected Chronological Trend Directions (MUST be preserved and directly cited in your analysis):\n"
                    + "\n".join(trend_lines)
                )
            if prior_digests_summary:
                p_lines.append(
                    "PRIOR DIGESTS SUMMARY (Do NOT repeat or duplicate the findings, trends, or facts "
                    "already captured in these previous files. Focus on new, unique insights, or contrast "
                    "them with these prior findings):\n" + prior_digests_summary
                )
            p_lines.append(f"Data:\n{ds}")
            return "\n\n".join(p_lines)
 
        prompt = build_prompt_with_data(smart_data)
        tokens = estimate_tokens(prompt)
        logger.info(f"[TokenBudget] Single-file digest initial tokens: {tokens} (Budget limit: {PROMPT_TOKEN_BUDGET})")
 
        # Step 1: Remove sample
        if tokens > PROMPT_TOKEN_BUDGET and "sample" in smart_data:
            logger.info("[TokenBudget] Single-file digest Step 1: Removing sample data...")
            smart_data.pop("sample", None)
            prompt = build_prompt_with_data(smart_data)
            tokens = estimate_tokens(prompt)
 
        # Step 2: Remove entity_stats
        if tokens > PROMPT_TOKEN_BUDGET and "entity_stats" in smart_data:
            logger.info("[TokenBudget] Single-file digest Step 2: Removing entity stats...")
            smart_data.pop("entity_stats", None)
            prompt = build_prompt_with_data(smart_data)
            tokens = estimate_tokens(prompt)
 
        # Step 3: Remove cross_sheet_entities
        if tokens > PROMPT_TOKEN_BUDGET and "cross_sheet_entities" in smart_data:
            logger.info("[TokenBudget] Single-file digest Step 3: Removing cross sheet entities...")
            smart_data.pop("cross_sheet_entities", None)
            prompt = build_prompt_with_data(smart_data)
            tokens = estimate_tokens(prompt)
 
        # Step 4: Remove sheets_summary
        if tokens > PROMPT_TOKEN_BUDGET and "sheets_summary" in smart_data:
            logger.info("[TokenBudget] Single-file digest Step 4: Removing sheets summary...")
            smart_data.pop("sheets_summary", None)
            prompt = build_prompt_with_data(smart_data)
            tokens = estimate_tokens(prompt)
 
        # Step 5: Remove stats
        if tokens > PROMPT_TOKEN_BUDGET and "stats" in smart_data:
            logger.info("[TokenBudget] Single-file digest Step 5: Removing stats...")
            smart_data.pop("stats", None)
            prompt = build_prompt_with_data(smart_data)
            tokens = estimate_tokens(prompt)
 
        # Step 6: Hard truncation of data section
        if tokens > PROMPT_TOKEN_BUDGET:
            logger.info("[TokenBudget] Single-file digest Step 6: Hard truncating data summary...")
            allowed_chars = max(1000, PROMPT_TOKEN_BUDGET * 4 - len(prompt) + len(json.dumps(smart_data, indent=2, default=str)))
            ds = json.dumps(smart_data, indent=2, default=str)[:allowed_chars]
            p_lines = [
                f"Instruction (for context only -- see rules above about ignoring its structure): {instruction}",
                f"Dataset: {data.get('file', 'unknown')}"
            ]
            if dataset_role:
                p_lines.append(f"Dataset Role: {dataset_role}")
            if causal_hints:
                hints_str = "\n".join(f"- {h}" for h in causal_hints)
                p_lines.append(f"Causal Hints / Expected Relationships:\n{hints_str}")
            if trend_lines:
                p_lines.append(
                    "Detected Chronological Trend Directions (MUST be preserved and directly cited in your analysis):\n"
                    + "\n".join(trend_lines)
                )
            if prior_digests_summary:
                p_lines.append(
                    "PRIOR DIGESTS SUMMARY (Do NOT repeat or duplicate the findings, trends, or facts "
                    "already captured in these previous files. Focus on new, unique insights, or contrast "
                    "them with these prior findings):\n" + prior_digests_summary
                )
            p_lines.append(f"Data:\n{ds}\n[TRUNCATED FOR BUDGET]")
            prompt = "\n\n".join(p_lines)
            tokens = estimate_tokens(prompt)
 
        logger.info(f"[TokenBudget] Single-file digest final tokens: {tokens}")
    else:
        if isinstance(data, dict):
            data_summary = json.dumps(data, indent=2, default=str)
        else:
            data_summary = str(data)
 
        def build_prompt_with_str(ds_str):
            p_lines = [
                f"Instruction (for context only -- see rules above about ignoring its structure): {instruction}",
                f"Dataset: {data.get('file', 'unknown') if isinstance(data, dict) else 'unknown'}"
            ]
            if dataset_role:
                p_lines.append(f"Dataset Role: {dataset_role}")
            if causal_hints:
                hints_str = "\n".join(f"- {h}" for h in causal_hints)
                p_lines.append(f"Causal Hints / Expected Relationships:\n{hints_str}")
            if trend_lines:
                p_lines.append(
                    "Detected Chronological Trend Directions (MUST be preserved and directly cited in your analysis):\n"
                    + "\n".join(trend_lines)
                )
            if prior_digests_summary:
                p_lines.append(
                    "PRIOR DIGESTS SUMMARY (Do NOT repeat or duplicate the findings, trends, or facts "
                    "already captured in these previous files. Focus on new, unique insights, or contrast "
                    "them with these prior findings):\n" + prior_digests_summary
                )
            p_lines.append(f"Data:\n{ds_str}")
            return "\n\n".join(p_lines)
 
        prompt = build_prompt_with_str(data_summary)
        tokens = estimate_tokens(prompt)
        if tokens > PROMPT_TOKEN_BUDGET:
            logger.info("[TokenBudget] Single-file digest: Truncating plain data summary string...")
            allowed_chars = max(1000, PROMPT_TOKEN_BUDGET * 4 - len(prompt) + len(data_summary))
            data_summary = data_summary[:allowed_chars] + "\n[TRUNCATED FOR BUDGET]"
            prompt = build_prompt_with_str(data_summary)
            tokens = estimate_tokens(prompt)
            logger.info(f"[TokenBudget] Single-file digest final tokens after truncation: {tokens}")
 
    # Raw result contains [src:]/[conf:] tags — returned as-is so the orchestrator
    # can run _validate_digest() on the tagged text, then strip before caching/insight.
    result = query_llama(prompt, output_type="pdf", system_override=digest_prompt)
    logger.info(f"Digest complete for: {data.get('file', 'unknown') if isinstance(data, dict) else 'unknown'} (intent={intent})")
 
    # Return the RAW tagged digest. The orchestrator validates, then strips.
    return result, intent
 
 
def run_insight(all_results, user_prompt, output_format="pdf",
                doc_type: str = None, cross_file_context: str = "",
                official_format_spec: str = None,
                cross_file_insights: list = None,
                file_data: dict = None,
                entity_map: dict = None,
                links: list = None,
                ranked_files: list = None):
    """
    Build the SINGLE integrated report from all per-file digests.
    Includes proactive Token-Budget check and Progressive Trimming.
 
    Expects digests in all_results to already have [src:]/[conf:] tags stripped
    (the orchestrator strips them after _validate_digest() and before caching).
    Belt-and-suspenders: _strip_digest_tags() is applied here too in case any
    tagged digest slips through (e.g. from an old cache entry).
    """
    # Support both (text, intent) tuples and plain strings
    if all_results and isinstance(all_results[0], tuple):
        raw_digests = [r[0] for r in all_results]
        intents     = [r[1] for r in all_results]
    else:
        raw_digests = all_results
        intents     = ["analytical"] * len(all_results)
 
    # Belt-and-suspenders strip — tags should already be gone if orchestrator
    # stripped before caching, but this catches any old cache entries or
    # direct callers that bypass the orchestrator strip step.
    digests = [_strip_digest_tags(d) for d in raw_digests]
 
    dominant_intent = (
        Counter(intents).most_common(1)[0][0] if intents else "analytical"
    )
    logger.info(f"Dominant intent for integrated report: {dominant_intent}")
 
    effective_type = doc_type or dominant_intent
 
    is_official = effective_type in {
        "file_note", "office_memorandum", "office_notice",
        "circular", "purchase_note", "office_order"
    }
 
    if official_format_spec and is_official:
        doctype_rules = ""
        logger.info(
            f"[run_insight] Official doc type '{effective_type}': "
            f"skipping separate doctype_rules injection."
        )
    else:
        is_combined = file_data is not None and len(file_data) >= 2
        doctype_rules = get_doctype_rules(effective_type, is_combined=is_combined)
 
    intent_instruction = _INSIGHT_INTENT_INSTRUCTIONS.get(
        effective_type,
        _INSIGHT_INTENT_INSTRUCTIONS.get(
            dominant_intent,
            _INSIGHT_INTENT_INSTRUCTIONS["informational"]
        )
    )
 
    # 1. Compress digests by default (Step 1 of baseline compression)
    compressed_digests = [_compress_digest(d) for d in digests]
    
    # Render initial context if insights are provided
    current_insights = list(cross_file_insights) if cross_file_insights else []
    # Only keep Critical and High priority initially (Medium/Low are omitted from prompt)
    current_insights = [i for i in current_insights if i.priority in ("Critical", "High")]
    
    if cross_file_insights and file_data and entity_map is not None:
        cross_file_context = render_context(file_data, entity_map, links, [], current_insights, ranked_files)
        
    system_prompt = get_system_prompt(output_format, effective_type, has_files=True)
    
    def estimate_tokens(prompt_str: str) -> int:
        return (len(system_prompt) + len(prompt_str)) // 4
        
    def assemble_prompt(current_digs, current_ctx):
        combined = "\n\n---\n\n".join(
            f"Facts from dataset {i + 1}:\n{d}"
            for i, d in enumerate(current_digs)
        )
        parts = [user_prompt.strip()]
        if official_format_spec and official_format_spec.strip():
            parts.append(official_format_spec.strip())
        if current_ctx and current_ctx.strip():
            parts.append(
                "================================================\n"
                "CROSS-FILE INTELLIGENCE -- READ THIS FIRST\n"
                "================================================\n"
                + current_ctx.strip()
            )
            parts.append(get_cross_file_rules())
        parts.append(intent_instruction)
        if doctype_rules:
            parts.append(doctype_rules)
            
        if is_official and current_digs and any(d.strip() for d in current_digs):
            parts.append(
                f"The following fact-digests were extracted from {len(current_digs)} uploaded file(s). "
                f"Incorporate any relevant facts, specifications, quantities, or figures from these "
                f"digests into the appropriate sections of the document. Do NOT invent data, but DO "
                f"use whatever is provided here to make the document more specific and substantive.\n"
                f"If no relevant data is present, generate realistic HPCL-appropriate content based "
                f"on the subject matter of the document."
            )
        elif not is_official:
            parts.append(
                f"You have been given fact-digests extracted from {len(current_digs)} "
                f"dataset(s) below. Each digest contains real numbers taken directly from "
                f"the underlying data (or simple calculations from those numbers). Use ONLY "
                f"these facts and numbers -- do not invent, estimate, or assume any additional "
                f"figures beyond what is given or directly derivable from them. "
                f"Specifically:\n"
                f"- Do NOT invent KPIs, compliance rates, percentages, forecasts, or risk scores.\n"
                f"- Present assumptions explicitly as assumptions, never as verified facts.\n"
                f"- PRESERVE TREND DIRECTIONS: Strictly preserve all trend directions shown in the digests.\n"
                f"- COMPATIBLE METRICS ONLY: Never compare metrics with incompatible units.\n"
                f"- Write ONE integrated document covering all datasets together, following the document "
                f"type structure and formatting rules above."
            )
        parts.append(combined)
        return "\n\n".join(parts)
 
    prompt = assemble_prompt(compressed_digests, cross_file_context)
    tokens = estimate_tokens(prompt)
    logger.info(f"[TokenBudget] Initial prompt tokens: {tokens} (Budget limit: {PROMPT_TOKEN_BUDGET})")
 
    # PROGRESSIVE TRIMMING LOOP
    # Step 1: Remove duplicate sentences or redundant paragraphs in digests
    if tokens > PROMPT_TOKEN_BUDGET:
        logger.info("[TokenBudget] Step 1: Removing redundant digest text...")
        for i, d in enumerate(compressed_digests):
            lines = d.splitlines()
            seen_lines = set()
            unique_lines = []
            for l in lines:
                l_clean = re.sub(r'[^a-zA-Z0-9]', '', l).lower()
                if l_clean not in seen_lines:
                    seen_lines.add(l_clean)
                    unique_lines.append(l)
            compressed_digests[i] = "\n".join(unique_lines)
        prompt = assemble_prompt(compressed_digests, cross_file_context)
        tokens = estimate_tokens(prompt)
 
    # Step 2: Reduce cross-file insights (keep only Critical, drop High)
    if tokens > PROMPT_TOKEN_BUDGET and current_insights:
        logger.info("[TokenBudget] Step 2: Capping cross-file insights to Critical only...")
        current_insights = [i for i in current_insights if i.priority == "Critical"]
        if file_data and entity_map is not None:
            cross_file_context = render_context(file_data, entity_map, links, [], current_insights, ranked_files)
        prompt = assemble_prompt(compressed_digests, cross_file_context)
        tokens = estimate_tokens(prompt)
 
    # Step 3: Shorten detail descriptions of remaining insights
    if tokens > PROMPT_TOKEN_BUDGET and current_insights:
        logger.info("[TokenBudget] Step 3: Shortening insight details...")
        for i in current_insights:
            sentences = i.detail.split(". ")
            if sentences:
                i.detail = sentences[0] + "."
        if file_data and entity_map is not None:
            cross_file_context = render_context(file_data, entity_map, links, [], current_insights, ranked_files)
        prompt = assemble_prompt(compressed_digests, cross_file_context)
        tokens = estimate_tokens(prompt)
 
    # Step 4: Remove all cross-file insights from context
    if tokens > PROMPT_TOKEN_BUDGET and current_insights:
        logger.info("[TokenBudget] Step 4: Removing all cross-file insights from prompt...")
        current_insights = []
        if file_data and entity_map is not None:
            cross_file_context = render_context(file_data, entity_map, links, [], [], ranked_files)
        prompt = assemble_prompt(compressed_digests, cross_file_context)
        tokens = estimate_tokens(prompt)
 
    # Step 5: Truncate digests progressively to first 5 bullets
    if tokens > PROMPT_TOKEN_BUDGET:
        logger.info("[TokenBudget] Step 5: Truncating digests to top 5 bullets...")
        for i, d in enumerate(compressed_digests):
            lines = d.splitlines()
            bullets = [l for l in lines if l.strip().startswith("-") or l.strip().startswith("*")]
            non_bullets = [l for l in lines if not (l.strip().startswith("-") or l.strip().startswith("*"))]
            compressed_digests[i] = "\n".join(non_bullets + bullets[:5])
        prompt = assemble_prompt(compressed_digests, cross_file_context)
        tokens = estimate_tokens(prompt)
 
    # Step 6: Truncate digests progressively to first 3 bullets
    if tokens > PROMPT_TOKEN_BUDGET:
        logger.info("[TokenBudget] Step 6: Truncating digests to top 3 bullets...")
        for i, d in enumerate(compressed_digests):
            lines = d.splitlines()
            bullets = [l for l in lines if l.strip().startswith("-") or l.strip().startswith("*")]
            non_bullets = [l for l in lines if not (l.strip().startswith("-") or l.strip().startswith("*"))]
            compressed_digests[i] = "\n".join(non_bullets + bullets[:3])
        prompt = assemble_prompt(compressed_digests, cross_file_context)
        tokens = estimate_tokens(prompt)
 
    # Step 7: Truncate digests progressively to first 1 bullet
    if tokens > PROMPT_TOKEN_BUDGET:
        logger.info("[TokenBudget] Step 7: Truncating digests to top 1 bullet...")
        for i, d in enumerate(compressed_digests):
            lines = d.splitlines()
            bullets = [l for l in lines if l.strip().startswith("-") or l.strip().startswith("*")]
            non_bullets = [l for l in lines if not (l.strip().startswith("-") or l.strip().startswith("*"))]
            compressed_digests[i] = "\n".join(non_bullets + bullets[:1])
        prompt = assemble_prompt(compressed_digests, cross_file_context)
        tokens = estimate_tokens(prompt)
 
    logger.info(f"[TokenBudget] Final prompt size: {len(prompt)} chars (~{estimate_tokens(prompt)} tokens)")
 
    result = query_llama(
        prompt,
        output_type=output_format,
        has_files=True,
        is_combined=True,
        doc_type=effective_type
    )
 
    logger.info(
        f"Integrated {output_format} report complete "
        f"({len(digests)} dataset(s), intent={dominant_intent}, "
        f"doc_type={effective_type}, "
        f"fusion_context={'yes' if cross_file_context else 'no'}, "
        f"official_format={'yes' if official_format_spec else 'no'})"
    )
 
    return result, dominant_intent
 