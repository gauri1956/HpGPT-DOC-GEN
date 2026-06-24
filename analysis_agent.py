import json
import logging
import re
from collections import Counter
 
from llm_agent import query_llama, get_doctype_rules, get_cross_file_rules
 
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
 
# -- Base digest prompt --------------------------------------------------------
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
- Where a chart adds value, insert it on its own line:
  [CHART: bar | title=Salary by Employee | labels=Rahul,Priya,Amit | values=65000,55000,72000 | y_label=Salary]
  Only chart numeric data actually present in the data.
- Do NOT generate charts for ID columns, serial numbers, or index columns.
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
 
 
def run_analysis(data, instruction, intent: str = None, dataset_role: str = None,
                 causal_hints: list = None):
    """
    Extract a fact-digest from a single dataset. Output is bullet points with
    real numbers (and optional chart markers) -- no report structure.
 
    Returns: (digest_text, intent)
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
    data_summary  = json.dumps(data, indent=2, default=str)[:5000]
 
    prompt_lines = [
        f"Instruction (for context only -- see rules above about ignoring its "
        f"structure): {instruction}",
        f"Dataset: {data.get('file', 'unknown')}"
    ]
 
    if dataset_role:
        prompt_lines.append(f"Dataset Role: {dataset_role}")
 
    if causal_hints:
        hints_str = "\n".join(f"- {h}" for h in causal_hints)
        prompt_lines.append(f"Causal Hints / Expected Relationships:\n{hints_str}")
 
    trend_lines = []
    if isinstance(data, dict):
        trends = data.get("trends", {})
        if trends:
            for col, t in trends.items():
                trend_lines.append(f"  - Column '{col}': {t.get('trend_signal', '')}")
 
    if trend_lines:
        prompt_lines.append(
            "Detected Chronological Trend Directions (MUST be preserved and directly cited in your analysis):\n"
            + "\n".join(trend_lines)
        )
 
    prompt_lines.append(f"Data:\n{data_summary}")
    prompt = "\n\n".join(prompt_lines)
 
    result = query_llama(prompt, output_type="pdf", system_override=digest_prompt)
    logger.info(f"Digest complete for: {data.get('file', 'unknown')} (intent={intent})")
 
    return result, intent
 
 
def run_insight(all_results, user_prompt, output_format="pdf",
                doc_type: str = None, cross_file_context: str = "",
                official_format_spec: str = None):
    """
    Build the SINGLE integrated report from all per-file digests.
 
    KEY FIX: When official_format_spec is present, doctype_rules are NOT
    injected separately -- doing so caused duplicate structure injection
    which confused the LLM and produced template-like output for Purchase Notes.
 
    Args:
        all_results          : list of (digest_text, intent) tuples from run_analysis()
        user_prompt          : original user request string
        output_format        : "docx" | "pdf" | "pptx"
        doc_type             : detected document type
        cross_file_context   : text block produced by DataFusionAgent
        official_format_spec : format spec from _resolve_official_format()
 
    Returns: (report_text, dominant_intent)
    """
    # Support both (text, intent) tuples and plain strings
    if all_results and isinstance(all_results[0], tuple):
        digests = [r[0] for r in all_results]
        intents = [r[1] for r in all_results]
    else:
        digests = all_results
        intents = ["analytical"] * len(all_results)
 
    dominant_intent = (
        Counter(intents).most_common(1)[0][0] if intents else "analytical"
    )
    logger.info(f"Dominant intent for integrated report: {dominant_intent}")
 
    effective_type = doc_type or dominant_intent
 
    # -- FIX: Only inject doctype_rules when official_format_spec is NOT present.
    # For official doc types, the format spec already contains the full structure.
    # Injecting doctype_rules ON TOP of official_format_spec causes the LLM to
    # see the same structure twice with minor wording differences -> confused output.
    is_official = effective_type in {
        "file_note", "office_memorandum", "office_notice",
        "circular", "purchase_note", "office_order"
    }
 
    if official_format_spec and is_official:
        # Official doc: use format spec ONLY -- no separate doctype_rules
        doctype_rules = ""
        logger.info(f"[run_insight] Official doc type '{effective_type}': skipping separate doctype_rules injection.")
    else:
        # Standard doc: use doctype_rules as usual
        doctype_rules = get_doctype_rules(effective_type)
 
    # Get narrative intent instruction
    intent_instruction = _INSIGHT_INTENT_INSTRUCTIONS.get(
        effective_type,
        _INSIGHT_INTENT_INSTRUCTIONS.get(
            dominant_intent,
            _INSIGHT_INTENT_INSTRUCTIONS["informational"]
        )
    )
 
    # Combine per-file digests
    combined = "\n\n---\n\n".join(
        f"Facts from dataset {i + 1}:\n{digest}"
        for i, digest in enumerate(digests)
    )[:6000]
 
    # -- Build the full prompt -------------------------------------------------
    # Structure:
    #   1. User request
    #   2. Official format spec (if present) -- MANDATORY structure for PSU doc types
    #   3. Cross-file intelligence context (if present)
    #   4. Cross-file rules (if context present)
    #   5. Intent instruction -- how to treat the data
    #   6. Doc-type structure rules -- ONLY if not official (to avoid duplication)
    #   7. Data grounding instructions
    #   8. Per-file digests
 
    prompt_parts = [user_prompt.strip()]
 
    # Inject official format spec FIRST so LLM sees the mandatory structure
    if official_format_spec and official_format_spec.strip():
        prompt_parts.append(official_format_spec.strip())
        logger.info(
            f"[run_insight] Official format spec injected "
            f"({len(official_format_spec)} chars) for doc_type='{effective_type}'."
        )
    else:
        logger.info(f"[run_insight] No official format spec -- standard generation.")
 
    if cross_file_context and cross_file_context.strip():
        prompt_parts.append(
            "================================================\n"
            "CROSS-FILE INTELLIGENCE -- READ THIS FIRST\n"
            "================================================\n"
            + cross_file_context.strip()
        )
        prompt_parts.append(get_cross_file_rules())
        logger.info(
            f"[run_insight] Cross-file context injected "
            f"({len(cross_file_context)} chars)."
        )
    else:
        logger.info("[run_insight] No cross-file context -- single-file or fusion skipped.")
 
    prompt_parts.append(intent_instruction)
 
    # Only inject doctype_rules for non-official docs
    if doctype_rules:
        prompt_parts.append(doctype_rules)
 
    # Data grounding instruction
    if is_official and digests and any(d.strip() for d in digests):
        # For official docs with uploaded files, tell LLM to use the file facts
        prompt_parts.append(
            f"The following fact-digests were extracted from {len(digests)} uploaded file(s). "
            f"Incorporate any relevant facts, specifications, quantities, or figures from these "
            f"digests into the appropriate sections of the document. Do NOT invent data, but DO "
            f"use whatever is provided here to make the document more specific and substantive.\n"
            f"If no relevant data is present, generate realistic HPCL-appropriate content based "
            f"on the subject matter of the document."
        )
    elif not is_official:
        prompt_parts.append(
            f"You have been given fact-digests extracted from {len(digests)} "
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
 
    prompt_parts.append(combined)
 
    prompt = "\n\n".join(prompt_parts)
 
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