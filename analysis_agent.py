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
}
 
 
def _build_digest_prompt(intent: str) -> str:
    """Combine base digest prompt with intent-specific instructions."""
    intent_block = _INTENT_INSTRUCTIONS.get(intent, _INTENT_INSTRUCTIONS["informational"])
    return _BASE_DIGEST_PROMPT.strip() + "\n\n" + intent_block.strip()
 
 
def run_analysis(data, instruction, intent: str = None):
    """
    Extract a fact-digest from a single dataset. Output is bullet points with
    real numbers (and optional chart markers) -- no report structure.
 
    intent: if not passed, detected automatically from data.
            detect_content_intent() returns (intent, confidence) -- we unpack it.
 
    Returns: (digest_text, intent)
    """
    from data_agent import detect_content_intent
 
    if isinstance(data, dict):
        clean_name = re.sub(r'^[a-f0-9]{8,}_', '', data.get('file', ''))
        clean_name = (clean_name.replace('.xlsx', '').replace('.csv', '')
                                .replace('_', ' ').strip())
        data = {**data, 'file': clean_name}
 
    if intent is None:
        # FIX: detect_content_intent now returns (intent, confidence) tuple
        intent, confidence = detect_content_intent(data, user_prompt=instruction)
        logger.info(
            f"Detected intent for {data.get('file', 'unknown')}: "
            f"{intent} (confidence={confidence})"
        )
    # If intent was passed in as a string by the orchestrator, use it directly
 
    digest_prompt = _build_digest_prompt(intent)
    data_summary  = json.dumps(data, indent=2, default=str)[:5000]
 
    prompt = (
        f"Instruction (for context only -- see rules above about ignoring its "
        f"structure): {instruction}\n\n"
        f"Dataset: {data.get('file', 'unknown')}\n\n"
        f"Data:\n{data_summary}"
    )
 
    result = query_llama(prompt, output_type="pdf", system_override=digest_prompt)
    logger.info(f"Digest complete for: {data.get('file', 'unknown')} (intent={intent})")
 
    return result, intent
 
 
def run_insight(all_results, user_prompt, output_format="pdf",
                doc_type: str = None, cross_file_context: str = ""):
    """
    Build the SINGLE integrated report from all per-file digests.
 
    Args:
        all_results        : list of (digest_text, intent) tuples from run_analysis(),
                             or list of plain strings (legacy fallback).
        user_prompt        : original user request string.
        output_format      : "docx" | "pdf" | "pptx"
        doc_type           : document type detected upstream (e.g. "business_report",
                             "training_material", "sop"). Overrides intent-based
                             output structure when provided.
        cross_file_context : text block produced by DataFusionAgent containing
                             shared entity links, confidence scores, numeric
                             summaries, and prioritised insights. When non-empty,
                             it is prepended to the LLM prompt so the model sees
                             cross-file intelligence BEFORE the per-file digests.
                             Also triggers injection of _CROSS_FILE_RULES into
                             the prompt so the LLM knows it MUST address them.
 
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
 
    # doc_type takes precedence for output structure; intent drives data treatment
    effective_type = doc_type or dominant_intent
 
    # Get output structure rules -- doc_type-specific if available, else intent-based
    doctype_rules = get_doctype_rules(effective_type)
 
    # Get narrative intent instruction for how to USE the data
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
    #   2. Cross-file intelligence context (if present) -- entity links,
    #      prioritised insights, shared dimensions, numeric summaries
    #   3. Cross-file rules (if context present) -- forces LLM to act on it
    #   4. Intent instruction -- how to treat the data
    #   5. Doc-type structure rules -- what sections to write
    #   6. Per-file digests -- the actual facts
 
    prompt_parts = [user_prompt.strip()]
 
    if cross_file_context and cross_file_context.strip():
        # Inject fusion context FIRST so the LLM sees relationships before facts
        prompt_parts.append(
            "================================================\n"
            "CROSS-FILE INTELLIGENCE -- READ THIS FIRST\n"
            "================================================\n"
            + cross_file_context.strip()
        )
        # Inject cross-file rules so the LLM knows it MUST use this context
        prompt_parts.append(get_cross_file_rules())
        logger.info(
            f"[run_insight] Cross-file context injected "
            f"({len(cross_file_context)} chars)."
        )
    else:
        logger.info("[run_insight] No cross-file context -- single-file or fusion skipped.")
 
    prompt_parts.append(intent_instruction)
    prompt_parts.append(doctype_rules)
 
    prompt_parts.append(
        f"You have been given fact-digests extracted from {len(digests)} "
        f"dataset(s) below. Each digest contains real numbers taken directly from "
        f"the underlying data (or simple calculations from those numbers). Use ONLY "
        f"these facts and numbers -- do not invent, estimate, or assume any additional "
        f"figures beyond what is given or directly derivable from them. Write ONE "
        f"integrated document covering all datasets together, following the document "
        f"type structure and formatting rules above."
    )
 
    prompt_parts.append(combined)
 
    prompt = "\n\n".join(prompt_parts)
 
    result = query_llama(
        prompt,
        output_type=output_format,
        has_files=True,
        is_combined=True
    )
 
    logger.info(
        f"Integrated {output_format} report complete "
        f"({len(digests)} dataset(s), intent={dominant_intent}, "
        f"doc_type={effective_type}, "
        f"fusion_context={'yes' if cross_file_context else 'no'})"
    )
 
    return result, dominant_intent
 