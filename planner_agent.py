import json
import logging
from llm_agent import query_llama
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Intent → document type routing table
# ---------------------------------------------------------------------------
# Maps detected content intent (from data_agent.detect_content_intent) to
# the most appropriate doc_type used downstream by analysis_agent / llm_agent.
# This ensures the planner locks the pipeline to the right prompt templates
# BEFORE any LLM calls happen, rather than relying on post-hoc detection.
# ---------------------------------------------------------------------------
_INTENT_DOCTYPE_MAP: dict[str, str] = {
    "analytical":    "business_report",
    "informational": "informational",
    "educational":   "training_material",
    "policy":        "policy_document",
    "operational":   "operational",
}
 
# Official PSU doc types detected from user prompt keywords.
# These bypass the intent-routing table entirely.
_OFFICIAL_DOC_KEYWORDS: dict[str, list[str]] = {
    "file_note":        ["file note", "file noting", "noting sheet"],
    "office_memorandum":["office memorandum", "o.m.", "om ", "inter-department memo",
                         "official memo"],
    "office_notice":    ["office notice", "notice board", "staff notice"],
    "circular":         ["circular", "office circular", "policy circular"],
    "purchase_note":    ["purchase note", "procurement note", "purchase order note",
                         "procurement justification"],
    "office_order":     ["office order", "transfer order", "posting order",
                         "sanction order"],
}
 
# Output format keyword detection from user prompt
_FORMAT_KEYWORDS: dict[str, list[str]] = {
    "pptx": ["ppt", "pptx", "presentation", "slide", "deck", "slides"],
    "pdf":  ["pdf", "report", "document"],
    "docx": ["word", "docx", "doc", "word document"],
}
 
 
# ---------------------------------------------------------------------------
# Helper: detect official doc type from prompt text
# ---------------------------------------------------------------------------
def _detect_official_doc_type(prompt: str) -> str | None:
    prompt_lower = prompt.lower()
    for doc_type, keywords in _OFFICIAL_DOC_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            logger.info(f"[planner] Official doc type detected from prompt: {doc_type}")
            return doc_type
    return None
 
 
# ---------------------------------------------------------------------------
# Helper: detect output format from prompt text
# ---------------------------------------------------------------------------
def _detect_output_format(prompt: str) -> str:
    prompt_lower = prompt.lower()
    for fmt, keywords in _FORMAT_KEYWORDS.items():
        if any(kw in prompt_lower for kw in keywords):
            return fmt
    return "pdf"  # default
 
 
# ---------------------------------------------------------------------------
# Helper: resolve doc_type from per-file intents
# ---------------------------------------------------------------------------
def _resolve_doc_type_from_intents(
    intents: list[str],
    official_doc_type: str | None,
    prompt: str,
) -> str:
    """
    Priority order:
    1. Official PSU doc type from prompt keywords (highest priority)
    2. Explicit analytical/financial keywords in prompt
    3. Most common intent across all files
    4. Fallback to "informational"
    """
    if official_doc_type:
        return official_doc_type
 
    prompt_lower = prompt.lower()
 
    # Explicit prompt overrides
    if any(w in prompt_lower for w in ["financial report", "finance report", "p&l", "profit and loss"]):
        return "financial_report"
    if any(w in prompt_lower for w in ["business report", "executive report", "kpi", "dashboard"]):
        return "business_report"
    if any(w in prompt_lower for w in ["sop", "standard operating procedure"]):
        return "sop"
 
    if not intents:
        return "informational"
 
    # Count intent votes; pick dominant
    from collections import Counter
    dominant = Counter(intents).most_common(1)[0][0]
    resolved = _INTENT_DOCTYPE_MAP.get(dominant, "informational")
 
    logger.info(
        f"[planner] Intent votes={Counter(intents)}, "
        f"dominant={dominant}, resolved doc_type={resolved}"
    )
    return resolved
 
 
# ---------------------------------------------------------------------------
# PLANNER_PROMPT  — produces structured JSON plan
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """
You are a document planning agent. Given a user request and a list of uploaded files,
your job is to break the task into clear subtasks and assign each to the right specialist agent.
 
Available agents:
- data_agent: reads and extracts structured data from a file (CSV, Excel, DOCX, PDF)
- analysis_agent: performs analysis on extracted data (trends, KPIs, comparisons, growth)
- graph_agent: generates charts from real data (bar, line, pie, horizontal_bar)
- insight_agent: generates recommendations, conclusions, cross-file comparisons
- summary_agent: summarizes content from a file
 
RULES:
- Every uploaded file must have exactly one data_agent task.
- graph_agent tasks are only needed for numerical/analytical datasets; skip for policy/text/educational files.
- insight_agent runs last and depends on ALL earlier tasks.
- Keep instruction text concise and specific to the file content.
 
Respond ONLY in this exact JSON format, no extra text:
{
  "document_title": "...",
  "subtasks": [
    {
      "task_id": "t1",
      "agent": "data_agent",
      "file": "filename.csv",
      "instruction": "Extract sales data including product names, quantities and revenue"
    },
    {
      "task_id": "t2",
      "agent": "analysis_agent",
      "depends_on": ["t1"],
      "instruction": "Analyze sales trends and identify top performing products"
    },
    {
      "task_id": "t3",
      "agent": "graph_agent",
      "depends_on": ["t1"],
      "instruction": "Generate bar chart of revenue by product"
    }
  ]
}
"""
 
 
# ---------------------------------------------------------------------------
# Public API: plan_tasks
# ---------------------------------------------------------------------------
def plan_tasks(
    prompt: str,
    file_names: list[str],
    file_intents: list[str] | None = None,
) -> dict:
    """
    Build an execution plan for the agent pipeline.
 
    Args:
        prompt       : User's original request string.
        file_names   : List of uploaded file names.
        file_intents : Optional list of pre-detected intents (one per file),
                       produced by data_agent.detect_content_intent().
                       When provided, the planner uses them to lock doc_type
                       without an extra LLM call.
 
    Returns:
        dict with keys:
            document_title  : str
            subtasks        : list[dict]
            doc_type        : str  ← NEW: locked document type for downstream agents
            output_format   : str  ← NEW: "docx" | "pdf" | "pptx"
            official_doc_type: str | None
    """
    # --- Step 1: Detect official doc type and output format from prompt ------
    official_doc_type = _detect_official_doc_type(prompt)
    output_format     = _detect_output_format(prompt)
 
    # --- Step 2: Resolve locked doc_type ------------------------------------
    doc_type = _resolve_doc_type_from_intents(
        intents           = file_intents or [],
        official_doc_type = official_doc_type,
        prompt            = prompt,
    )
 
    # --- Step 3: Call LLM for task decomposition ----------------------------
    user_msg = f"User request: {prompt}\n\nUploaded files: {', '.join(file_names)}"
    raw = query_llama(user_msg, output_type="plan", system_override=PLANNER_PROMPT)
 
    plan = _parse_plan(raw, prompt, file_names)
 
    # --- Step 4: Inject resolved metadata into plan -------------------------
    plan["doc_type"]          = doc_type
    plan["output_format"]     = output_format
    plan["official_doc_type"] = official_doc_type
 
    # --- Step 5: Validate and repair subtasks --------------------------------
    plan["subtasks"] = _validate_subtasks(plan["subtasks"], file_names, doc_type)
 
    logger.info(
        f"[planner] Plan ready: {len(plan['subtasks'])} subtasks, "
        f"doc_type={doc_type}, output_format={output_format}, "
        f"official={official_doc_type}"
    )
    return plan
 
 
# ---------------------------------------------------------------------------
# Internal: parse LLM JSON response
# ---------------------------------------------------------------------------
def _parse_plan(raw: str, prompt: str, file_names: list[str]) -> dict:
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM response")
        plan = json.loads(raw[start:end])
        logger.info(f"[planner] LLM plan parsed: {len(plan.get('subtasks', []))} subtasks")
        return plan
    except Exception as e:
        logger.error(f"[planner] Failed to parse LLM plan JSON: {e}\nRaw: {raw[:500]}")
        return _fallback_plan(prompt, file_names)
 
 
# ---------------------------------------------------------------------------
# Internal: validate and repair subtask list
# ---------------------------------------------------------------------------
def _validate_subtasks(
    subtasks: list[dict],
    file_names: list[str],
    doc_type: str,
) -> list[dict]:
    """
    Ensure structural integrity of the subtask list:
 
    1. Every uploaded file has at least one data_agent task.
    2. graph_agent tasks are removed for non-analytical doc types to avoid
       LLM wasting tokens on chart generation for policy/text documents.
    3. Every task has a valid task_id.
    4. insight_agent task exists and depends on all others.
 
    Returns a repaired subtask list.
    """
    if not subtasks:
        return _fallback_plan("", file_names)["subtasks"]
 
    # --- Rule 1: Ensure every file has a data_agent task --------------------
    covered_files = {
        t.get("file", "")
        for t in subtasks
        if t.get("agent") == "data_agent"
    }
    tid_counter = max(
        (int(t.get("task_id", "t0").lstrip("t")) for t in subtasks),
        default=0
    ) + 1
 
    for fname in file_names:
        if fname not in covered_files:
            logger.warning(
                f"[planner] File '{fname}' had no data_agent task — adding one."
            )
            subtasks.insert(0, {
                "task_id":    f"t{tid_counter}",
                "agent":      "data_agent",
                "file":       fname,
                "instruction": f"Extract all structured data from {fname}",
            })
            tid_counter += 1
 
    # --- Rule 2: Remove graph_agent tasks for non-analytical docs -----------
    _NO_CHART_TYPES = {
        "file_note", "office_memorandum", "office_notice",
        "circular", "purchase_note", "office_order",
        "policy_document", "sop", "training_material", "informational",
        "educational", "policy",
    }
    if doc_type in _NO_CHART_TYPES:
        removed = [t for t in subtasks if t.get("agent") == "graph_agent"]
        if removed:
            logger.info(
                f"[planner] Removing {len(removed)} graph_agent task(s) "
                f"— doc_type '{doc_type}' does not use charts."
            )
        subtasks = [t for t in subtasks if t.get("agent") != "graph_agent"]
 
    # --- Rule 3: Ensure insight_agent exists and has full dependencies ------
    non_insight_ids = [
        t["task_id"] for t in subtasks
        if t.get("agent") != "insight_agent" and "task_id" in t
    ]
    insight_tasks = [t for t in subtasks if t.get("agent") == "insight_agent"]
 
    if not insight_tasks:
        logger.info("[planner] No insight_agent task found — adding one.")
        subtasks.append({
            "task_id":    f"t{tid_counter}",
            "agent":      "insight_agent",
            "depends_on": non_insight_ids,
            "instruction": "Generate integrated insights and final report content.",
        })
    else:
        # Update existing insight_agent to depend on all non-insight tasks
        for t in subtasks:
            if t.get("agent") == "insight_agent":
                existing_deps = set(t.get("depends_on", []))
                t["depends_on"] = list(existing_deps | set(non_insight_ids))
 
    return subtasks
 
 
# ---------------------------------------------------------------------------
# Fallback plan (used when LLM JSON parse fails)
# ---------------------------------------------------------------------------
def _fallback_plan(prompt: str, file_names: list[str]) -> dict:
    """
    Generate a deterministic fallback plan without any LLM call.
    Produces one data_agent + one analysis_agent + one graph_agent per file,
    then a single insight_agent at the end.
    """
    subtasks = []
    tid = 1
    data_task_ids = []
 
    for f in file_names:
        data_tid = f"t{tid}"
        subtasks.append({
            "task_id":    data_tid,
            "agent":      "data_agent",
            "file":       f,
            "instruction": f"Extract all structured data from {f}",
        })
        data_task_ids.append(data_tid)
        tid += 1
 
        subtasks.append({
            "task_id":    f"t{tid}",
            "agent":      "analysis_agent",
            "depends_on": [data_tid],
            "instruction": f"Analyse data from {f} and identify key insights and trends",
        })
        tid += 1
 
        subtasks.append({
            "task_id":    f"t{tid}",
            "agent":      "graph_agent",
            "depends_on": [data_tid],
            "instruction": f"Generate relevant charts from numerical data in {f}",
        })
        tid += 1
 
    all_ids = [t["task_id"] for t in subtasks]
    subtasks.append({
        "task_id":    f"t{tid}",
        "agent":      "insight_agent",
        "depends_on": all_ids,
        "instruction": prompt or "Generate integrated cross-file insights and final report.",
    })
 
    logger.warning("[planner] Using fallback plan (LLM parse failed).")
    return {
        "document_title": "Analysis Report",
        "subtasks":       subtasks,
        # These get overwritten by plan_tasks() after _fallback_plan returns,
        # but set sensible defaults here for safety.
        "doc_type":           "business_report",
        "output_format":      "pdf",
        "official_doc_type":  None,
    }
 