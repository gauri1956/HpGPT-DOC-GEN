import os
import re
import time
import random
import requests
from dotenv import load_dotenv
import logging

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is missing. "
        "Please check your .env file or configure GROQ_API_KEY in your environment."
    )
GROQ_API_URL   = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME     = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

# ── Token budget ──────────────────────────────────────────────────────────────
# PROMPT_TOKEN_BUDGET is exported so analysis_agent and agent_orchestrator can
# gate their own prompt assembly before calling query_llama.
# llm_agent itself never trims or rewrites the prompt it receives.
MODEL_CONTEXT_LIMIT    = 8192
RESERVED_OUTPUT_TOKENS = 2500
SYSTEM_PROMPT_OVERHEAD = 900   # conservative estimate for assembled system prompt
PROMPT_TOKEN_BUDGET    = MODEL_CONTEXT_LIMIT - RESERVED_OUTPUT_TOKENS - SYSTEM_PROMPT_OVERHEAD
# = 4792 tokens available for the USER message (digests + context + instructions)

# ── Chart marker guide ────────────────────────────────────────────────────────
CHART_MARKER_GUIDE = """
When your content would benefit from a chart, insert a marker on its own line:
[CHART: bar | title=Revenue by Quarter | labels=Q1,Q2,Q3,Q4 | values=12.5,18.2,15.0,22.1 | y_label=Revenue (M)]
[CHART: line | title=Growth Trend | labels=Jan,Feb,Mar | values=100,120,140 | y_label=Units]
[CHART: pie | title=Market Share | labels=HP,Dell,Lenovo | values=35,28,37]
[CHART: horizontal_bar | title=Team Performance | labels=Sales,Support,Dev | values=88,92,76 | x_label=Score]

Rules: place on its own line; labels and values must match in count; values are plain numbers only;
only insert where a chart genuinely adds value; do NOT describe the chart in text.
"""

_CHART_COVERAGE_RULES = (
    "\nCHART COVERAGE:\n"
    "- Insert charts for analytical/operational/financial/KPI datasets with meaningful numeric data.\n"
    "- Charts are optional for educational, policy, SOP, and training documents.\n"
    "- Prefer line charts for time-series, bar/horizontal_bar for categorical comparisons.\n"
)

# ── Core rules (all prompts) ──────────────────────────────────────────────────
_CORE_RULES = (
    "Use organization-specific branding (e.g. HPCL) only when explicitly present in source files or requested.\n"
    "Every sentence must be complete. Do not include placeholder text like '[insert data here]' or '[TBD]'.\n"
    "Start your response with a # title.\n"
    "Do NOT invent corporate examples, case studies, or specific facts unless explicitly in the source.\n"
    "NEVER wrap dataset/file/column names in quotes.\n"
    "Derive the document title from the source material, not from generic labels.\n"
    "HEADING ACCURACY: Every ## or ### heading must accurately label ALL content that follows it until the next heading.\n"
    "NUMBER FORMATTING: Round all numeric values to at most ONE decimal place. Whole numbers need no decimal point.\n"
    "NO REPETITION: State each fact, statistic, or finding ONLY ONCE in the entire document.\n"
    "LANGUAGE VARIETY: Vary sentence openers; do not overuse 'could indicate', 'indicating a', 'this suggests'.\n"
)

# ── Grounding rules (file-based prompts only) ─────────────────────────────────
_GROUNDING_RULES = (
    "\nDOCUMENT-TYPE ADAPTATION:\n"
    "Apply analytical/business-impact reasoning ONLY for business, financial, KPI, or operational reports.\n"
    "For educational, policy, compliance, and SOP documents: focus on explanation, procedures, and knowledge transfer.\n"

    "\nDATA GROUNDING:\n"
    "Before writing ANY section, scan ALL dataset summaries -- every file, every column, every stat block.\n"
    "NEVER write that a topic is 'not available' unless you have checked every dataset and found no related column.\n"
    "Every dataset provided must be referenced by at least one specific statistic.\n"

    "\nNO HALLUCINATION:\n"
    "NEVER invent Reference Numbers, Employee Names, Budgets, Vendors, KPI targets, or Dates unless explicitly provided.\n"
    "For required but missing fields: use underlines (e.g. 'Ref No.: __________________') or draft indicators.\n"
    "Recommendations must be qualitative or based strictly on calculated evidence -- never invent target percentages.\n"

    "\nSPECIFICITY:\n"
    "Every recommendation must name a SPECIFIC metric, entity, and its actual value from the data.\n"
    "Forbidden generic phrases: 'improve efficiency', 'increase sales', 'optimize inventory', 'enhance performance'.\n"
    "NEVER state a numeric KPI target, compliance rate, or forecast unless it is in the input data or a direct calculation.\n"

    "\nCONSISTENCY:\n"
    "PRESERVE all trend directions (up/down/stable) from data digests -- do not contradict them.\n"
    "NEVER compare metrics with incompatible units or dimensions.\n"

    "\nANALYTICAL DEPTH (for business/financial/operational reports):\n"
    "For every key finding: Observed Metric -> Cross-File Evidence -> Causal Hypothesis -> Business Impact -> Recommendation.\n"
    "Executive Summary must answer: (1) Most important finding? (2) Why? (3) One action leadership must take.\n"
    "Identify the single largest gap/trend/risk across ALL datasets as the headline insight.\n"
    "Do NOT treat each dataset as a mini-report -- connect findings causally across datasets.\n"
    "Cross-dataset correlations must come from pre-computed cross-file intelligence, not invented relationships.\n"
)

# ── Doc-type-specific spec rules ──────────────────────────────────────────────
_REPORT_SPECIFIC_RULES = (
    "\nREPORT FINDINGS FORMAT:\n"
    "Structure each finding as:\n"
    "  * **Observation**: [what happened]\n"
    "  * **Evidence**: [exact numbers/values]\n"
    "  * **Why it matters**: [operational/business impact]\n"
    "  * **Recommended action**: [specific qualitative next step]\n"
    "When stating correlations, include the Pearson r value and Insight Strength label as given in context.\n"
    "Include a Strategic Recommendations Table: "
    "| Recommendation | Operational Rationale | Expected Outcome & Metrics | Priority | Owner | Timeline |\n"
)

_TRAINING_SPECIFIC_RULES = (
    "\nTRAINING CONTENT RULES:\n"
    "Focus on instructional/educational/onboarding content with clear learning stages.\n"
    "NEVER use report labels: 'Observation', 'Evidence', 'Why it matters', 'Recommended action', "
    "'Insight Strength', 'Pearson correlation'.\n"
)

_SOP_SPECIFIC_RULES = (
    "\nSOP CONTENT RULES:\n"
    "Focus on standard operating procedures, roles, timelines, and sequential process steps.\n"
    "NEVER use report labels: 'Observation', 'Evidence', 'Why it matters', 'Recommended action', 'Insight Strength'.\n"
)

# ── Multi-file rules ──────────────────────────────────────────────────────────
_MULTIFILE_RULES = (
    "\nMULTI-FILE RULES:\n"
    "1. Write ONE unified report covering ALL files together.\n"
    "2. Use each section heading EXACTLY ONCE.\n"
    "3. Do NOT write a separate block per file or use file names as section headings.\n"
    "4. Compare and synthesize insights across files.\n"
    "5. ONE consolidated Recommendations section at the very end.\n"
    "6. Do NOT add a 'Charts & Visualisations' section at the end.\n"
)

# ── Cross-file rules ──────────────────────────────────────────────────────────
_CROSS_FILE_RULES = (
    "\nCROSS-FILE INTELLIGENCE:\n"
    "- Address every CRITICAL and HIGH priority insight listed.\n"
    "- Use shared entity dimensions to compare values across files.\n"
    "- Executive Summary headline must come from cross-file analysis when multiple files exist.\n"
    "- Use connecting language (e.g. 'Taken alongside...', 'This is compounded by...') backed by evidence.\n"
)

# ── No-file data rules ────────────────────────────────────────────────────────
_NOINPUT_DATA_RULES = (
    "\nNO-FILE GENERATION RULES:\n"
    "No source data has been uploaded. Generate a structurally correct document framework.\n"
    "PRIORITY: (1) Uploaded data, (2) Explicit prompt info, (3) Placeholders, (4) Never invent facts.\n"
    "- Use underlines (__________________) for all unknown dates, ref numbers, names, and costs.\n"
    "- Do NOT invent: dates, reference numbers, committee names, KPI values, specific costs, or product names.\n"
    "- Use 'Ref No.: __________________', 'Date: __________________', 'Estimated Cost: Rs. __________________'.\n"
    "- NEVER use words like 'estimated 95%', 'hypothetical 95%', 'illustrative 95%'.\n"
    "- Issuing authority must be HPCL (or relevant HPCL department) unless otherwise specified.\n"
    "- Include '## Assumptions & Disclaimer' ONLY for: business_report, policy_document, sop, training_material.\n"
    "  Disclaimer text: 'This document provides a template/framework. Organization-specific information "
    "must be supplied by the concerned department.'\n"
    "- Do NOT add Assumptions & Disclaimer to: office_memorandum, circular, office_order, file_note, purchase_note.\n"
    "- For business_report with no files: write a structured template with placeholder fields, NOT smooth narrative prose.\n"
    "- For purchase_note with no files: use underlines for all specs, quantities, costs, timelines.\n"
    "- For placeholders with numeric ranges: write '______ to ______' not 'from to'.\n"
)

# ── Document-type output rules ────────────────────────────────────────────────
_DOCTYPE_OUTPUT_RULES = {

    "file_note": (
        "\nDOCUMENT TYPE -- FILE NOTE:\n"
        "  # File Note: [Subject]\n"
        "  File No.: __________________\n"
        "  Department: __________________\n"
        "  Date: __________________\n"
        "  Subject: [subject]\n"
        "  ---\n"
        "  ## Background / Facts of the Case\n"
        "  ## Analysis / Deliberations\n"
        "  ## Financial Implications  (omit if not applicable)\n"
        "  ## Recommendation\n"
        "  ## Approval Sought\n"
        "  Submitted for Approval:\n"
        "  Prepared by / Designation / Department: __________________\n"
        "  Recommended by / Designation / Department: __________________\n"
        "  Approved by (Competent Authority) / Designation / Date: __________________\n"
        "CRITICAL RULES: Formal government language. Number every paragraph sequentially (1., 2., 3.).\n"
        "Reference prior correspondence as 'vide letter no.'. NO bullet points in body -- use numbered paragraphs.\n"
        "Do NOT add sections not listed. Do NOT generate [CHART:] markers. Use underlines for unknown fields.\n"
        "Full signatory chain is MANDATORY.\n"
    ),

    "office_memorandum": (
        "\nDOCUMENT TYPE -- OFFICE MEMORANDUM (O.M.):\n"
        "  # Office Memorandum\n"
        "  No.: __________________          Date: __________________\n"
        "  From: [Issuing Office / Department]\n"
        "  To:   [Recipient Office / Department]\n"
        "  Subject: [Subject]\n"
        "  ---\n"
        "  ## Body\n"
        "    (Open with 'The undersigned is directed to...' -- numbered paragraphs -- directive content ONLY)\n"
        "    (If data files uploaded: convert findings to actionable directives -- NO raw statistics or column names)\n"
        "  ## Action Required  (omit if not applicable)\n"
        "  Sd/-\n"
        "  Name / Designation / Department: __________________\n"
        "CRITICAL RULES: Open with 'The undersigned is directed to...' -- NEVER first-person 'I'.\n"
        "Numbered paragraphs throughout. Body is DIRECTIVE/INFORMATIONAL -- not an analytical report.\n"
        "Do NOT add Executive Summary or KPI sections. Do NOT generate [CHART:] markers. Use underlines for unknowns.\n"
    ),

    "office_notice": (
        "\nDOCUMENT TYPE -- OFFICE NOTICE:\n"
        "  # Office Notice\n"
        "  Notice No.: __________________             Date: __________________\n"
        "  Department: __________________\n"
        "  Subject: [subject]\n"
        "  ---\n"
        "  ## Notice Body\n"
        "    (Open with 'It is hereby notified that...' -- numbered points)\n"
        "  ## Compliance / Action Required\n"
        "  By Order / For [Issuing Authority]\n"
        "  Signature / Name / Designation / Department: __________________\n"
        "  Distribution: 1. All Concerned Departments  2. Notice Board  3. Office Copy\n"
        "CRITICAL RULES: Numbered points in body. State deadline and responsible party in Compliance.\n"
        "No analytical language. Do NOT generate [CHART:] markers. Use underlines for unknowns.\n"
    ),

    "circular": (
        "\nDOCUMENT TYPE -- OFFICE CIRCULAR:\n"
        "  # Circular No. __________________\n"
        "  Date: __________________\n"
        "  Subject: [subject]\n"
        "  ---\n"
        "  ## Preamble / Background\n"
        "  ## Scope & Applicability  (must name employee categories: permanent, contractual, temporary)\n"
        "  ## Instructions / Directives  (MINIMUM 6 numbered directives)\n"
        "  ## Effective Date\n"
        "  ## Non-Compliance  (omit if not applicable)\n"
        "  By Order of [Authority]\n"
        "  Signature / Name / Designation / Department: __________________\n"
        "  Distribution: 1. All Departments  2. IT Division (for portal hosting)  3. Office Copy\n"
        "CRITICAL RULES: Numbered instructions -- NO prose paragraphs for directives.\n"
        "Preamble MUST reference a specific regulation or HPCL policy. Effective date must be stated.\n"
        "Do NOT generate [CHART:] markers. Use underlines for unknowns.\n"
    ),

    "purchase_note": (
        "\nDOCUMENT TYPE -- PURCHASE / PROCUREMENT NOTE:\n"
        "A justification and decision-support document for procurement approval. NOT a payment form.\n"
        "  # Purchase Note: [Item or Service]\n"
        "  Note No.: __________________          Date: __________________\n"
        "  Department: __________________\n"
        "  Subject: Purchase of [description]\n"
        "  ---\n"
        "  ## 1. Purpose and Operational Requirement\n"
        "    3-5 numbered paragraphs: what is procured and why; operational gap it fills;\n"
        "    which dept/process uses it; current situation without it; alignment with HPCL objectives.\n"
        "  ## 2. Specification of Items / Services\n"
        "    | S.No | Item / Service Description | Technical Specification | Quantity | Est. Unit Cost (Rs.) | Total Est. Cost (Rs.) |\n"
        "    Use 'Rs. __________________' for unknown costs.\n"
        "  ## 3. Operational Justification and Risk of Non-Procurement\n"
        "    3-5 numbered paragraphs: why these specs; alternatives considered; specific risks if not procured.\n"
        "  ## 4. Budget Provision and Financial Considerations\n"
        "    2-3 numbered paragraphs: budget head; sanctioned/sought status; estimated total cost.\n"
        "  ## 5. Procurement Method and Vendor Strategy\n"
        "    2-3 numbered paragraphs: method (Open Tender/Limited Tender/Single Source/GeM/Rate Contract);\n"
        "    rationale; GFR 2017 / HPCL procurement guideline reference.\n"
        "  ## 6. Proposed Delivery Schedule\n"
        "    1-2 numbered paragraphs: expected timeline; milestones.\n"
        "  ## 7. Recommendation and Approval Sought\n"
        "    Formal recommendation paragraph + exact sanction being sought.\n"
        "    Prepared by / Recommended by / Approved by (Competent Authority): __________________\n"
        "CRITICAL RULES: Sections 1 and 3 MUST have 3-5 numbered paragraphs each -- never one-liners.\n"
        "Section 2 MUST be a properly formatted markdown table with all 6 columns.\n"
        "Section 5 MUST name the procurement method with GFR 2017 reference.\n"
        "Use numbered paragraphs (1., 2., 3.) in every section -- NO bullet points.\n"
        "Use underlines (__________________) for unknowns -- NEVER brackets like [TBD].\n"
        "Do NOT generate [CHART:] markers. Full signatory chain MANDATORY.\n"
        "Specs and justifications must be SPECIFIC to the item requested -- not generic business benefits.\n"
    ),

    "office_order": (
        "\nDOCUMENT TYPE -- OFFICE ORDER:\n"
        "  # Office Order No. __________________\n"
        "  Date: __________________\n"
        "  Subject: [subject]\n"
        "  ---\n"
        "  ## Order\n"
        "    (Open with 'It is ordered that...' -- MINIMUM 3 numbered paragraphs)\n"
        "  ## Effective Date\n"
        "  ## Compliance  (name responsible officer/dept and reporting frequency)\n"
        "  By Order of Competent Authority\n"
        "  Signature / Name / Designation / Department: __________________\n"
        "  Copy to: 1. All Concerned Officers  2. HR Division / Personal File  3. Office Copy\n"
        "CRITICAL RULES: Open with 'It is ordered that...' -- numbered paragraphs throughout.\n"
        "Effective date and compliance requirements must be explicit.\n"
        "Do NOT generate [CHART:] markers. Use underlines for unknowns.\n"
    ),

    "training_material": (
        "\nDOCUMENT TYPE -- TRAINING MATERIAL:\n"
        "  ## Learning Objectives (4-6 measurable objectives)\n"
        "  ## [Topic Sections -- one per major concept]\n"
        "  ## Assessment (5 questions with answers based only on source content)\n"
        "  ## Glossary (key terms with definitions)\n"
        "  ## Key Takeaways\n"
        "Do NOT create fictional HPCL scenarios or KPI dashboards.\n"
    ),

    "policy_document": (
        "\nDOCUMENT TYPE -- POLICY / COMPLIANCE DOCUMENT:\n"
        "  ## Purpose & Scope\n"
        "  ## Applicability\n"
        "  ## Key Requirements\n"
        "  ## Roles & Responsibilities\n"
        "  ## Procedures / Implementation Steps\n"
        "  ## Non-Compliance Consequences\n"
        "  ## Compliance Checklist\n"
        "For every requirement: state WHAT, WHO is responsible, WHEN, and consequences.\n"
    ),

    "sop": (
        "\nDOCUMENT TYPE -- STANDARD OPERATING PROCEDURE (SOP):\n"
        "  ## Purpose & Scope\n"
        "  ## Prerequisites\n"
        "  ## Procedure (numbered steps: Action -> Role -> Tool -> Safety Note)\n"
        "  ## Critical Control Points\n"
        "  ## Quality Checks\n"
        "  ## Safety Warnings\n"
        "  ## Troubleshooting (symptom -> cause -> corrective action)\n"
        "  ## Performance Metrics\n"
    ),

    "business_report": (
        "\nDOCUMENT TYPE -- BUSINESS / ANALYTICAL REPORT:\n"
        "  ## Executive Summary  (headline insight -> root cause -> 90-day action)\n"
        "  ## KPI Dashboard  (table: Metric | Value | Trend; add Target/Variance only if targets are in the data)\n"
        "  ## Key Findings  (each finding: Observation / Evidence / Why it matters / Recommended action)\n"
        "  ## [Topic-specific analysis sections]\n"
        "  ## Risk Assessment  (table: Risk | Likelihood | Impact | Mitigation)\n"
        "  ## Strategic Recommendations  (table: Recommendation | Operational Rationale | Expected Outcome & Metrics)\n"
    ),

    "financial_report": (
        "\nDOCUMENT TYPE -- FINANCIAL REPORT:\n"
        "  ## Executive Summary\n"
        "  ## Financial Snapshot\n"
        "  ## Variance Analysis and Business Impact\n"
        "  ## [Segment analysis sections]\n"
        "  ## Risk Assessment\n"
        "  ## Forward Outlook\n"
        "  ## Priority Actions\n"
    ),

    "operational": (
        "\nDOCUMENT TYPE -- OPERATIONAL STATUS REPORT:\n"
        "  ## Operational Summary\n"
        "  ## Category-wise Breakdown\n"
        "  ## Anomalies & Flags\n"
        "  ## Operational Recommendations\n"
        "  ## Performance Metrics\n"
    ),

    "informational": (
        "\nDOCUMENT TYPE -- INFORMATIONAL DOCUMENT:\n"
        "  ## Introduction\n"
        "  ## [Topic sections derived from content]\n"
        "  ## Summary\n"
        "Present information accurately. Do NOT invent analysis, KPIs, or recommendations.\n"
    ),

    "educational": (
        "\nDOCUMENT TYPE -- EDUCATIONAL CONTENT:\n"
        "Structure: Introduction -> topic-by-topic sections -> Key Takeaways.\n"
        "Do NOT reframe as a KPI dashboard or HR analytics report.\n"
    ),

    "policy": (
        "\nDOCUMENT TYPE -- POLICY DOCUMENT:\n"
        "Structure: Purpose & Scope -> Requirements -> Procedures -> Responsibilities -> Compliance.\n"
        "Present faithfully -- no invented analysis.\n"
    ),
}

# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {

    "docx": (
        "You are an expert enterprise document generator. Customize branding and terminology "
        "to match the organization and domain of the input files or user request. "
        "If the context is explicitly HPCL, use HPCL branding; otherwise remain domain-agnostic.\n\n"
        "FORMATTING:\n"
        "- Use ## for section headings, ### for subsections.\n"
        "- Use bullet points (- ) for lists and markdown tables for structured data.\n"
        "- Formal corporate tone. Minimum 6 sections with at least 3 sentences of real analysis each.\n"
    ),

    "pdf": (
        "You are an expert enterprise PDF report generator. Customize branding to match "
        "the organization and domain of the input files or user request.\n\n"
        "FORMATTING:\n"
        "- Use ## for major sections, ### for subsections.\n"
        "- Formal corporate tone. Each section must have at least 3 sentences of real analysis.\n"
    ),

    "pptx": (
        "You are an expert PowerPoint presentation designer. Enforce strict slide-ready discipline:\n"
        "1. SLIDE STRUCTURE: Use ## for each slide heading. Each ## becomes exactly one slide.\n"
        "2. BULLETS: Exactly 3-5 bullet points per slide using '-' syntax.\n"
        "3. LENGTH: Each bullet = one direct sentence, max 18 words. No sub-bullets.\n"
        "4. CHARTS: One chart marker per slide where numeric data exists.\n"
        "5. SPEAKER NOTES: Every slide MUST have: 'Speaker Notes: <detailed paragraph>'\n"
        "6. NO WRAPPER SLIDES: Never write ## Title Page, ## Acknowledgement, or ## Thank You.\n"
        "7. SLIDE COUNT: 10-14 slides total, each covering a distinct topic.\n"
    ),

    "official_doc": (
        "You are an expert in drafting official government and PSU documents. "
        "If the context is explicitly HPCL, draft for HPCL; otherwise match the domain of the request.\n\n"
        "IDENTITY: You are acting as a senior officer drafting an official document. "
        "Your output must read as a real, complete, professionally written government document.\n\n"
        "CONTENT RULES:\n"
        "1. Generate SUBSTANTIVE CONTENT for every section -- no one-line sections.\n"
        "2. Purchase Notes S1 and S3: each needs 3-5 numbered paragraphs of real operational reasoning.\n"
        "3. File Notes: Background and Analysis sections need 3+ numbered paragraphs.\n"
        "4. Circulars/Notices: Instructions must be a numbered list of minimum 6 specific directives.\n"
        "5. Derive content from context given. 'Purchase note for laptops' -> real operational justification.\n"
        "6. FULL SIGNATORY CHAIN MANDATORY for all official document types.\n\n"
        "ANTI-HALLUCINATION (metadata fields only):\n"
        "- NEVER write real or fictional person names. Use: __________________\n"
        "- NEVER invent reference numbers. Use: Reference No.: __________________\n"
        "- NEVER write specific dates unless given. Use: Date: __________________\n"
        "- NEVER invent financial figures. Use: Rs. __________________\n\n"
        "FORMATTING:\n"
        "- Follow the EXACT format structure provided in the user prompt.\n"
        "- Formal, impersonal government language. NEVER use first-person 'I'. Use 'the undersigned'.\n"
        "- Number all paragraphs within body sections sequentially (1., 2., 3.).\n"
        "- Do NOT add sections not specified. Do NOT generate charts or KPI dashboards.\n"
        "- Start with a # heading containing the document type and subject.\n"
    ),

    "docx_noinput": (
        "You are an expert enterprise document writer creating a document from scratch with no source data.\n\n"
        "FORMATTING:\n"
        "- If required by disclaimer rules, first section MUST be ## Assumptions & Disclaimer.\n"
        "- Use ## for headings, ### for subsections. Minimum 6 sections with meaningful content.\n"
        "- Do NOT generate [CHART:] markers -- no data to chart.\n"
    ),

    "pdf_noinput": (
        "You are an expert enterprise PDF writer creating a document from scratch with no source data.\n\n"
        "FORMATTING:\n"
        "- If required by disclaimer rules, first section MUST be ## Assumptions & Disclaimer.\n"
        "- Use ## for major sections, ### for subsections. Minimum 6 sections.\n"
        "- Do NOT generate [CHART:] markers -- no data to chart.\n"
    ),

    "pptx_noinput": (
        "You are an expert PowerPoint presentation designer creating slides from scratch:\n"
        "1. SLIDE STRUCTURE: Use ## for each slide heading. Each ## becomes exactly one slide.\n"
        "2. BULLETS: Exactly 3-5 bullet points per slide using '-' syntax.\n"
        "3. LENGTH: Each bullet = one direct sentence, max 18 words.\n"
        "4. SPEAKER NOTES: Every slide MUST have: 'Speaker Notes: <detailed paragraph>'\n"
        "5. NO WRAPPER SLIDES: Never write ## Title Page, ## Acknowledgement, or ## Thank You.\n"
        "6. SLIDE COUNT: 10-14 slides total.\n"
    ),

    "plan": (
        "You are a document title generator. Output a single 5-7 word professional document title.\n"
        "Rules: Output ONLY the title -- no quotes, no punctuation, no preamble.\n"
        "Derive the title from the ACTUAL FILE CONTENT. Never produce generic titles.\n"
        "Only include 'HPCL' if the content is specifically about HPCL operations.\n"
        "Use correct casing: 'HPCL', 'HR', 'FY', 'KPI' in full uppercase.\n"
    ),
}

_OFFICIAL_DOC_TYPES = {
    "file_note", "office_memorandum", "office_notice",
    "circular", "purchase_note", "office_order"
}


def get_system_prompt(output_type: str, doc_type: str = None, has_files: bool = True) -> str:
    """
    Assemble the system prompt for a given output type and document type.
    Content/grounding rules are appended here; the caller owns user-message content.
    """
    if output_type == "plan":
        return SYSTEM_PROMPTS["plan"]

    if doc_type in _OFFICIAL_DOC_TYPES:
        base = SYSTEM_PROMPTS["official_doc"] + "\n\n" + _CORE_RULES
        if not has_files:
            base += "\n\n" + _NOINPUT_DATA_RULES
        return base

    spec_rules = ""
    is_analytical = doc_type in ("business_report", "financial_report", "operational", "analytical")
    if is_analytical:
        spec_rules = _REPORT_SPECIFIC_RULES
    elif doc_type == "training_material":
        spec_rules = _TRAINING_SPECIFIC_RULES
    elif doc_type == "sop":
        spec_rules = _SOP_SPECIFIC_RULES

    if not has_files:
        key  = f"{output_type}_noinput"
        base = SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS.get(output_type, SYSTEM_PROMPTS["docx"]))
        return base + "\n\n" + _CORE_RULES + spec_rules + "\n\n" + _NOINPUT_DATA_RULES

    base  = SYSTEM_PROMPTS.get(output_type, SYSTEM_PROMPTS["docx"])
    rules = _CORE_RULES + "\n\n" + _GROUNDING_RULES + spec_rules + "\n\n" + _MULTIFILE_RULES

    if is_analytical and output_type in ("docx", "pdf"):
        rules += "\n\n" + _CHART_COVERAGE_RULES + "\n" + CHART_MARKER_GUIDE
    elif is_analytical and output_type == "pptx":
        rules += "\n\nFor analytical reports, distribute charts across slides where meaningful numeric data exists.\n"

    return base + "\n\n" + rules


def get_doctype_rules(doc_type: str, is_combined: bool = False) -> str:
    rules = _DOCTYPE_OUTPUT_RULES.get(doc_type, _DOCTYPE_OUTPUT_RULES["informational"])
    if is_combined and doc_type == "business_report":
        if "## Risk Assessment" in rules:
            rules = rules.replace(
                "## Risk Assessment",
                "## Cross-File Synthesis & Efficiency Comparisons\n  ## Risk Assessment"
            )
        if "Expected Outcome & Metrics)\n" in rules:
            rules = rules.replace(
                "Expected Outcome & Metrics)\n",
                "Expected Outcome & Metrics)\n  "
                "## Implementation Roadmap  (Immediate 30d / Short-term 90d / Medium-term 180d)\n"
            )
    return rules


def get_cross_file_rules() -> str:
    return _CROSS_FILE_RULES


def is_official_doc_type(doc_type: str) -> bool:
    return doc_type in _OFFICIAL_DOC_TYPES


def _get_max_tokens(doc_type: str, is_combined: bool, output_type: str) -> int:
    """
    Choose how many output tokens to request.
    This is the only place max_tokens is decided -- callers do not override it.
    """
    if doc_type and doc_type in _OFFICIAL_DOC_TYPES:
        return 3500   # official docs need full multi-section content
    if is_combined:
        return 3500   # integrated multi-file report
    if output_type == "pptx":
        return 1536   # slides are compact
    return 2048       # single-file digest or simple document


def query_llama(prompt: str, output_type: str = "docx",
                is_combined: bool = False,
                has_files: bool = True,
                system_override: str = None,
                doc_type: str = None) -> str:
    """
    Call the Groq LLaMA model with exponential backoff for rate limits.

    Single responsibility: HTTP transport + retry logic.

    What this function does:
      - Assembles the payload (system prompt + user message).
      - Chooses max_tokens via _get_max_tokens().
      - Retries on 429 / 503 with backoff; falls back to FALLBACK_MODEL.
      - On 413: reduces max_tokens to shrink output reservation and retries.
        It does NOT trim or alter the prompt -- that is the caller's responsibility.

    The caller (analysis_agent.run_analysis / analysis_agent.run_insight /
    agent_orchestrator) is responsible for ensuring the prompt already fits
    within PROMPT_TOKEN_BUDGET before this function is called.
    """
    if system_override:
        system_prompt = system_override
    else:
        system_prompt = get_system_prompt(output_type, doc_type, has_files)
        logging.info(
            f"[query_llama] output_type='{output_type}', "
            f"doc_type='{doc_type}', has_files={has_files}"
        )

    max_tokens = _get_max_tokens(doc_type, is_combined, output_type)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    models_to_try = [MODEL_NAME, FALLBACK_MODEL]

    for model in models_to_try:
        current_max_tokens = max_tokens   # may be halved on 413 within this model's attempts

        for attempt in range(4):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0.4,
                    "max_tokens":  current_max_tokens,
                }

                logging.info(
                    f"Groq call: model={model}, attempt={attempt + 1}, "
                    f"output_type={output_type}, doc_type={doc_type}, "
                    f"max_tokens={current_max_tokens}, "
                    f"prompt_chars={len(prompt)}"
                )

                response = requests.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=90,
                )

                # ── 413 Payload Too Large ─────────────────────────────────
                # Only reduce max_tokens -- do NOT alter the prompt.
                # If reducing max_tokens doesn't help, escalate to next model.
                # Log a clear message so the caller knows to reduce prompt size.
                if response.status_code == 413:
                    logging.error(
                        f"Groq API 413 on {model} attempt {attempt + 1}. "
                        f"prompt_chars={len(prompt)}, max_tokens={current_max_tokens}. "
                        f"Reducing max_tokens. If this persists, reduce prompt size in caller."
                    )
                    if current_max_tokens > 512:
                        current_max_tokens = max(512, current_max_tokens // 2)
                        time.sleep(2)
                        continue
                    else:
                        logging.error(
                            f"413 on {model} even at min max_tokens={current_max_tokens}. "
                            f"Trying next model. Caller must reduce prompt size."
                        )
                        break

                # ── 429 Rate Limit ────────────────────────────────────────
                if response.status_code == 429:
                    retry_after = (
                        response.headers.get("Retry-After")
                        or response.headers.get("x-ratelimit-reset-requests")
                        or response.headers.get("x-ratelimit-reset")
                    )
                    server_wait = None
                    if retry_after:
                        try:
                            raw  = str(retry_after)
                            nums = re.findall(r"\d+\.?\d*", raw)
                            if nums:
                                val = float(nums[0])
                                if "ms" in raw.lower():
                                    val = val / 1000.0
                                server_wait = val
                        except Exception:
                            pass

                    base_wait = min(15 * (2 ** attempt), 90)
                    jitter    = random.uniform(0, 5)
                    wait      = max(server_wait or 0, base_wait) + jitter

                    if model != models_to_try[-1] and wait > 15:
                        logging.warning(
                            f"Rate limited (429) on {model}. "
                            f"Wait {wait:.1f}s too long -- switching to fallback model."
                        )
                        break

                    wait = min(wait, 90.0)
                    logging.warning(
                        f"Rate limited (429) on {model}, attempt {attempt + 1}. "
                        f"server_wait={server_wait}, computed_wait={wait:.1f}s. Sleeping..."
                    )
                    time.sleep(wait)
                    continue

                # ── 503 Service Unavailable ───────────────────────────────
                if response.status_code == 503:
                    wait = 10 * (attempt + 1) + random.uniform(0, 3)
                    logging.warning(
                        f"Service unavailable (503) on {model}, "
                        f"attempt {attempt + 1}. Waiting {wait:.1f}s."
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                result = response.json()["choices"][0]["message"]["content"].strip()

                min_len = 10 if output_type == "plan" else 50
                if not result or len(result) < min_len:
                    logging.warning(
                        f"Very short response from {model} "
                        f"(len={len(result) if result else 0}). Retrying."
                    )
                    time.sleep(5)
                    continue

                logging.info(f"Groq response: {len(result)} chars from {model}")
                return result

            except requests.exceptions.Timeout:
                logging.warning(f"Timeout on {model} attempt {attempt + 1}. Retrying.")
                time.sleep(8)

            except requests.exceptions.ConnectionError:
                logging.warning(f"Connection error on {model} attempt {attempt + 1}. Retrying.")
                time.sleep(8)

            except requests.exceptions.HTTPError as e:
                logging.warning(f"HTTP error on {model} attempt {attempt + 1}: {e}")
                time.sleep(5)

            except Exception as e:
                logging.warning(f"Unexpected error on {model} attempt {attempt + 1}: {e}")
                time.sleep(5)

        logging.warning(f"All attempts failed for {model}, trying next model.")

    raise Exception(
        "Groq API failed after all retries on all models. "
        "Check logs for rate limit or connectivity details."
    )

