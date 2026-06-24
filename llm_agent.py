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
 
# -- Chart marker guide --------------------------------------------------------
CHART_MARKER_GUIDE = """
When your content would benefit from a chart, insert a marker on its own line in EXACTLY this format:
[CHART: bar | title=Revenue by Quarter | labels=Q1,Q2,Q3,Q4 | values=12.5,18.2,15.0,22.1 | y_label=Revenue (M)]
[CHART: line | title=Growth Trend | labels=Jan,Feb,Mar | values=100,120,140 | y_label=Units]
[CHART: pie | title=Market Share | labels=HP,Dell,Lenovo | values=35,28,37]
[CHART: horizontal_bar | title=Team Performance | labels=Sales,Support,Dev | values=88,92,76 | x_label=Score]
 
STRICT rules for chart markers:
- Place the marker on its own line, never inside a sentence or paragraph
- labels and values must have the same count
- Values must be plain numbers only -- no %, $, commas, or units inside values
- Only insert a marker where a chart genuinely adds value
- Do NOT describe the chart in text -- the chart speaks for itself
- Do NOT write [CHART: ...] inside bullet points or sentences
- Place the chart marker IMMEDIATELY after the paragraph it supports -- not at the end of the document
"""
 
# -- Chart coverage rules ------------------------------------------------------
_CHART_COVERAGE_RULES = (
    "\nCHART COVERAGE -- CRITICAL:\n"
    "- For analytical, operational, financial, and KPI datasets, insert charts whenever meaningful numeric data exists.\n"
    "- For educational, informational, policy, compliance, SOP, and training documents, charts are OPTIONAL.\n"
    "- Do NOT create charts solely to satisfy formatting requirements.\n"
    "- If no meaningful numeric data exists, do not generate chart markers.\n"
    "- If a dataset has a time/period column paired with a numeric column, prefer a line chart.\n"
    "- If a dataset has a categorical column paired with a numeric column, prefer a bar or horizontal_bar chart.\n"
)
 
# -- Core rules -- applied to ALL prompts --------------------------------------
_BASE_CORE_RULES = (
    "Use organization-specific branding (like HPCL) only when explicitly present in the source files or requested by the user.\n"
    "Every sentence must be complete -- never end mid-word or mid-thought.\n"
    "Do not include placeholder text like '[insert data here]' or '[TBD]'.\n"
    "Start your response with a # title.\n"
)

# -- Report core rules -- applied only to report, training, or analytical documents
_REPORT_CORE_RULES = (
    "Do NOT invent corporate examples, case studies, scenarios, or specific facts for HPCL or any other organization unless they are explicitly present in the source document or explicitly requested by the user.\n"
    "NEVER wrap dataset names, file names, or column names in quotes.\n"
    "For uploaded documents, derive the title primarily from the source material and document content.\n"
    "Do not generate generic titles such as Employee Training Program, Development Program, KPI Dashboard, Learning Module, or Training Presentation unless those phrases explicitly appear in the source material.\n"
    "\n"
    "HEADING ACCURACY -- CRITICAL:\n"
    "- Every ## or ### heading must be an accurate label for ALL the content\n"
    "  that follows it until the next heading.\n"
    "- Before writing content under a heading, check: does every sentence/bullet\n"
    "  match what this heading promises If a point belongs to a different topic,\n"
    "  move it to the correct section or rename the heading.\n"
    "- Never use a generic heading as a catch-all for leftover points.\n"
    "\n"
    "NUMBER FORMATTING -- CRITICAL:\n"
    "- Round all numeric values in sentences to at most ONE decimal place.\n"
    "- If a number is a whole number, write it with NO decimal point at all.\n"
    "- Never write more than one decimal place.\n"
    "\n"
    "NO REPETITION -- CRITICAL:\n"
    "- State each fact, statistic, or finding ONLY ONCE in the entire document.\n"
    "- Do not restate the same number or insight in a later section.\n"
    "\n"
    "LANGUAGE VARIETY -- CRITICAL:\n"
    "- Do not overuse phrases like 'could indicate', 'indicating a', 'this suggests'.\n"
    "- Vary sentence openers and verbs across the whole document.\n"
    "\n"
    "CONSULTANT-GRADE FINDINGS -- CRITICAL:\n"
    "- For every finding or key insight in the report, you MUST structure it using this format:\n"
    "  * **Observation**: [Brief sentence describing what happened]\n"
    "  * **Evidence**: [The exact numbers, values, or percentage growths from the data]\n"
    "  * **Why it matters**: [The operational/business impact or risk]\n"
    "  * **Recommended action**: [The specific qualitative next step]\n"
    "- When stating any correlation or trend, you MUST explicitly output the Pearson correlation coefficient (r = X.XX) and confidence/strength label (Insight Strength: [Very High/High/Medium]) as given in the prompt context. Never speak vaguely about correlations without providing these exact values.\n"
)

_CORE_RULES = _BASE_CORE_RULES + _REPORT_CORE_RULES
 
# -- Grounding + analytical reasoning rules ------------------------------------
_BASE_GROUNDING_RULES = (
    "\nDOCUMENT-TYPE ADAPTATION -- CRITICAL:\n"
    "- Apply analytical, profitability, competitive-position, revenue-impact, board-level, and strategic-action reasoning ONLY for business, financial, KPI, or operational reports.\n"
    "- For educational, informational, policy, compliance, and SOP documents, focus on accurate explanation, responsibilities, procedures, requirements, and knowledge transfer.\n"
    "- Do NOT force business-impact analysis onto documents whose purpose is instructional, procedural, or regulatory.\n"

    "\nDATA GROUNDING -- CRITICAL:\n"
    "- Before writing ANY section, scan ALL dataset summaries provided in the user\n"
    "  message -- every file, every column, every stat block -- at least once.\n"
    "- NEVER write that a topic is 'not included', 'not available', or 'not in the\n"
    "  data' unless you have checked every dataset and found genuinely no related\n"
    "  column or value. If even one related number exists, use it.\n"
    "- Every dataset provided must be referenced by at least one specific statistic\n"
    "  somewhere in the document.\n"

    "\nNO HALLUCINATED REFERENCES or METRIC FABRICATION -- CRITICAL:\n"
    "- The system should NEVER invent Reference Numbers, File Numbers, Employee Names, Budgets/Costs, specific Vendors, KPI targets, or Dates unless they are explicitly provided by the user or found in the uploaded files.\n"
    "- If a reference number, date, or officer name is required by the document format but not provided, you MUST use standard professional blank underlines (e.g., 'Ref No.: __________________', 'Date: __________________', 'Officer Name: __________________') or draft indicators (e.g. '[Draft Reference]', '[Responsible Officer]'). Never invent realistic-looking placeholder data.\n"
    "- Do NOT fabricate numbers, target KPIs, or names of external vendors. Use general terms (e.g., 'vendor', 'authorized supplier', 'target threshold') instead of fictional names/values.\n"
    "- NEVER invent target percentages, numeric reductions, cost savings, or revenue growth figures in the Recommendations section (e.g., 'reduce inventory by 10%' or 'save INR 50 million') unless these specific numbers are explicitly derived from calculations or present in the source files. Recommendations must be qualitative (e.g., 'Review inventory policy for Diesel') or based strictly on calculated evidence.\n"

    "\n-- SPECIFICITY -- no generic consulting filler -- CRITICAL:\n"
    "- Every recommendation must name a SPECIFIC metric, entity, and its actual value, gap, or trend from the provided data. Generic or generic-industry filler is strictly forbidden.\n"
    "- The following generic recommendations are STRICTLY FORBIDDEN: 'improve efficiency', 'increase sales', 'optimize inventory', 'enhance performance', 'strengthen compliance', 'maximize productivity', 'streamline operations'. Instead, you must write recommendations directly tied to data findings with specific metrics.\n"

    "\nNO INVENTED METRICS, TARGETS, OR FORECASTS -- CRITICAL:\n"
    "- NEVER state a specific numeric value, KPI target, compliance rate, variance, or forecast percentage unless it is either:\n"
    "  (a) given directly in the input dataset summaries, OR\n"
    "  (b) a simple arithmetic derivation of numbers that ARE in the dataset.\n"

    "\nCONSISTENCY CHECK -- CRITICAL:\n"
    "- Do not write generic findings unless they are TRUE for the specific numbers provided.\n"
    "\nCOMPATIBLE METRIC COMPARISONS -- CRITICAL:\n"
    "- NEVER compare or combine metrics with incompatible units or dimensions.\n"
    "\nPRESERVE TREND SIGNALS -- CRITICAL:\n"
    "- You MUST strictly preserve all trend directions (up, down, stable/flat) identified in the data digests and cross-file insights.\n"
)

_ANALYTICAL_RULES = (
    "\n-- CONTENT PRIORITIZATION & ROADMAPS -- CRITICAL:\n"
    "- Identify the 3-5 most important concepts or findings. Give them the most coverage.\n"
    "- Minor details must NOT receive equal weight to critical findings.\n"
    "- Compliance, safety, operational risk, and strategic items always rank above background.\n"
    "- You MUST include a structured 'Implementation Roadmap' table at the end of the recommendations section with columns: | Action Item | Priority (Critical/High/Medium) | Responsible Function | Timeline (30 / 60 / 90 Days) | Target Success Metric |.\n"

    "\n-- DEPTH OVER SUMMARY & ANALYSIS OVER DESCRIPTION -- CRITICAL:\n"
    "- Do NOT merely list or describe facts. For every key finding: state it -> explain WHY it matters -> describe its operational/business impact -> state what action follows.\n"
    "- Every major section must answer: What happened? Why? Why does it matter? What risks exist? What must be done?\n"

    "\n-- STRATEGIC THINKING & SECTOR CONTEXT -- CRITICAL:\n"
    "- Apply sector-specific business reasoning ONLY when the provided datasets or user request explicitly reference the relevant context.\n"
    "- Every section must connect its findings to a business consequence relevant to the actual domain of the data.\n"

    "\n-- ROOT-CAUSE ANALYSIS -- CRITICAL:\n"
    "- Never stop at the symptom. For every problem identified, write one sentence naming the most likely underlying cause.\n"

    "\n-- CROSS-DATASET CORRELATION -- CRITICAL:\n"
    "- Actively look for relationships ACROSS datasets when meaningful.\n"
    "- Only claim relationships when supported by direct evidence in the cross-file intelligence or data.\n"

    "\n-- EXECUTIVE STORYTELLING -- CRITICAL:\n"
    "- The document must have a narrative arc: situation -> complication -> insight -> recommendation.\n"
    "- The Executive Summary must answer: (1) What is the single most important thing happening? (2) Why is it happening? (3) What is the one action leadership must take?\n"

    "\n-- HEADLINE INSIGHT -- CRITICAL:\n"
    "- Identify the single largest gap, trend, risk, or opportunity visible across ALL provided datasets.\n"
    "- This must be the FIRST substantive statement in the Executive Summary.\n"

    "\nINTEGRATED NARRATIVE -- CRITICAL:\n"
    "- Do not treat each dataset as its own mini-report. Connect findings ACROSS datasets causally.\n"
)

_GROUNDING_RULES = _BASE_GROUNDING_RULES + _ANALYTICAL_RULES
_SHARED_RULES = _CORE_RULES + _GROUNDING_RULES
 
# -- Data rules for NOINPUT prompts -------------------------------------------
_NOINPUT_DATA_RULES = (
    "\nDATA RULES FOR NO-FILE / ILLUSTRATIVE REPORTS -- CRITICAL:\n"
    "- No source data file has been uploaded. You are generating an illustrative document based on assumptions.\n"
    "- ALL numerical values, KPIs, compliance rates, percentages, forecasts, and risk scores you generate MUST be explicitly marked as hypothetical estimates.\n"
    "- You MUST include a prominent 'Assumptions & Disclaimer' section at the very beginning stating all metrics are hypothetical.\n"
    "- Do NOT leave any unresolved placeholder values. All placeholders must be resolved to illustrative scenario values.\n"
)
 
# -- Multi-file rules ----------------------------------------------------------
_MULTIFILE_RULES = (
    "\nMULTI-FILE STRUCTURE RULES -- CRITICAL, always enforced:\n"
    "1. Write ONE unified report covering ALL uploaded files together.\n"
    "2. Use each section heading EXACTLY ONCE across the entire document.\n"
    "3. Do NOT write a separate section block per file.\n"
    "4. Do NOT use file names as section headings.\n"
    "5. Compare and synthesize insights across files when meaningful.\n"
    "6. The Recommendations section must be ONE consolidated section at the very end.\n"
    "7. NEVER write 'The data can be visualized as follows:' as standalone content.\n"
    "8. Do NOT add a 'Charts & Visualisations' section at the end.\n"
)
 
# -- Cross-file intelligence rules ---------------------------------------------
_CROSS_FILE_RULES = (
    "\nCROSS-FILE INTELLIGENCE -- CRITICAL:\n"
    "- A cross-file intelligence block is provided above the digests.\n"
    "- You MUST explicitly address every CRITICAL and HIGH priority insight listed.\n"
    "- Use the shared entity dimensions to compare values across files when relevant.\n"
    "- The Executive Summary headline insight must come from the cross-file analysis when multiple files exist.\n"
    "- Use connecting language (e.g. 'Taken alongside...', 'This is compounded by...') when supported by actual evidence.\n"
)
 
# -- Document-type-aware output instructions -----------------------------------
_DOCTYPE_OUTPUT_RULES = {
 
    "file_note": (
        "\nDOCUMENT TYPE -- FILE NOTE (Official PSU/Government Format):\n"
        "A File Note records deliberations, facts, and recommendations for decision-making.\n"
        "Structure MUST follow this exact order:\n"
        "  # File Note: [Subject]\n"
        "  File No.: __________________\n"
        "  Department: __________________\n"
        "  Date: __________________\n"
        "  Subject: [subject in brief]\n"
        "  ---\n"
        "  ## Background / Facts of the Case\n"
        "    (Chronological facts, previous correspondence references, context)\n"
        "  ## Analysis / Deliberations\n"
        "    (Point-wise examination of options, pros/cons, regulatory references)\n"
        "  ## Financial Implications\n"
        "    (Budget head, estimated cost, approval required -- omit section if not applicable)\n"
        "  ## Recommendation\n"
        "    (Clear, specific recommendation -- one or two paragraphs)\n"
        "  ## Approval Sought\n"
        "    (State exactly what approval/sanction is being sought)\n\n"
        "  Submitted for Approval:\n\n"
        "  Prepared by: __________________\n"
        "  Designation: __________________\n"
        "  Department: __________________\n\n"
        "  Recommended by: __________________\n"
        "  Designation: __________________\n"
        "  Department: __________________\n\n"
        "  Approved by (Competent Authority): __________________\n"
        "  Designation: __________________\n"
        "  Date: __________________\n"
        "CRITICAL RULES:\n"
        "- Use formal government language throughout.\n"
        "- Number every paragraph inside each ## section sequentially (1., 2., 3. ...).\n"
        "- Reference file numbers and prior correspondence with 'vide letter no.' format.\n"
        "- End with a clear 'Approval Sought' statement -- never leave it open-ended.\n"
        "- Do NOT use bullet points inside body sections -- use numbered paragraphs.\n"
        "- Do NOT add sections not listed above.\n"
        "- Do NOT generate [CHART: ...] markers in this document type.\n"
        "- Use underlines (________________) for any field values not provided in context -- NEVER use brackets.\n"
        "- FULL SIGNATORY CHAIN MANDATORY: Render Prepared by, Recommended by, and Approved by (Competent Authority) blocks completely.\n"
        "- SPECIFIC SYSTEM RECOMMENDATIONS: Recommendations must be specific to the proposed system rather than generic.\n"
    ),
 
    "office_memorandum": (
        "\nDOCUMENT TYPE -- OFFICE MEMORANDUM / O.M. (Official PSU/Government Format):\n"
        "An Office Memorandum is used for inter-departmental official communication.\n"
        "Structure MUST follow this exact order:\n"
        "  # Office Memorandum\n"
        "  No.: __________________          Date: __________________\n"
        "  From: [Issuing Office / Department]\n"
        "  To:   [Recipient Office / Department]\n"
        "  Subject: [Subject]\n"
        "  ---\n"
        "  ## Body\n"
        "    (Open with 'The undersigned is directed to...' or reference to prior O.M.)\n"
        "    (Content in numbered paragraphs -- directive/notification content ONLY)\n"
        "    (If files were uploaded, summarise the KEY DIRECTIVE or FINDING in plain\n"
        "     language suitable for an inter-departmental memo -- NOT raw statistics\n"
        "     or correlation coefficients. Convert data findings into actionable directives.)\n"
        "  ## Action Required\n"
        "    (What the recipient must do and by when -- omit if not applicable)\n\n"
        "  Sd/-\n"
        "  __________________\n"
        "  Name: __________________\n"
        "  Designation: __________________\n"
        "  Department / Division: __________________\n"
        "CRITICAL RULES:\n"
        "- Open with 'The undersigned is directed to...' -- NEVER use first-person 'I'.\n"
        "- Use numbered paragraphs throughout the Body section.\n"
        "- Body content must be DIRECTIVE and INFORMATIONAL -- not analytical reports.\n"
        "- If data files are provided, convert findings into actionable memo language.\n"
        "  DO NOT paste raw statistics, correlation coefficients, or dataset column names.\n"
        "  INSTEAD write: 'The sales data for the period indicates a significant upward trend\n"
        "  in the Northern region, requiring immediate attention from regional managers.'\n"
        "- Do NOT add Executive Summary, KPI Dashboard, or analytical report sections.\n"
        "- Do NOT generate [CHART: ...] markers in this document type.\n"
        "- Use underlines (________________) for any field values not provided in context.\n"
    ),
 
    "office_notice": (
        "\nDOCUMENT TYPE -- OFFICE NOTICE (Official PSU/Government Format):\n"
        "An Office Notice communicates information or directives to a group of recipients.\n"
        "Structure MUST follow this exact order:\n"
        "  # Office Notice\n"
        "  Notice No.: __________________             Date: __________________\n"
        "  Department: __________________\n"
        "  Subject: [subject]\n"
        "  ---\n"
        "  ## Notice Body\n"
        "    (Open with 'It is hereby notified that...' or 'All concerned are informed that...')\n"
        "    (Key information in clear numbered points)\n"
        "  ## Compliance / Action Required\n"
        "    (What recipients must do, by when)\n\n"
        "  By Order / For [Issuing Authority]\n\n"
        "  Signature: __________________\n"
        "  Name: __________________\n"
        "  Designation: __________________\n"
        "  Department: __________________\n\n"
        "  Distribution:\n"
        "  1. All Concerned Departments\n"
        "  2. Notice Board\n"
        "  3. Office Copy\n"
        "CRITICAL RULES:\n"
        "- Begin with 'It is hereby notified that...' or 'All concerned are informed that...'\n"
        "- Use numbered points inside sections.\n"
        "- State deadline and responsible party explicitly in the Compliance section.\n"
        "- Do NOT use analytical or report-style language.\n"
        "- Do NOT generate [CHART: ...] markers in this document type.\n"
        "- Use underlines (________________) for any field values not provided in context.\n"
    ),
 
    "circular": (
        "\nDOCUMENT TYPE -- OFFICE CIRCULAR (Official PSU/Government Format):\n"
        "A Circular is a policy or procedure communication issued to a wide audience.\n"
        "Structure MUST follow this exact order:\n"
        "  # Circular No. __________________\n"
        "  Date: __________________\n"
        "  Subject: [subject]\n"
        "  ---\n"
        "  ## Preamble / Background\n"
        "    (Why this circular is issued; reference to policy/authority under which issued)\n"
        "  ## Scope & Applicability\n"
        "    (Who this circular applies to)\n"
        "  ## Instructions / Directives\n"
        "    (Numbered list of specific instructions -- each instruction on its own numbered line)\n"
        "  ## Effective Date\n"
        "    (State the date from which this circular is effective)\n"
        "  ## Non-Compliance\n"
        "    (Consequences of non-compliance -- omit if not applicable)\n\n"
        "  By Order of [Authority]\n\n"
        "  Signature: __________________\n"
        "  Name: __________________\n"
        "  Designation: __________________\n"
        "  Department: __________________\n\n"
        "  Distribution:\n"
        "  1. All Departments / Sections\n"
        "  2. IT Division (for hosting on Portal)\n"
        "  3. Office Copy\n"
        "CRITICAL RULES:\n"
        "- Use numbered instructions -- NEVER prose paragraphs for directives.\n"
        "- State the effective date explicitly.\n"
        "- Reference the authority/policy under which this circular is issued.\n"
        "- The Preamble must reference a specific regulation, ministry guideline, or HPCL policy.\n"
        "- The Instructions section must have MINIMUM 6 specific, actionable numbered directives.\n"
        "- Do NOT generate [CHART: ...] markers in this document type.\n"
        "- Use underlines (________________) for any field values not provided in context.\n"
        "- SCOPE section must explicitly name employee categories (permanent, contractual, temporary).\n"
    ),
 
    "purchase_note": (
        "\nDOCUMENT TYPE -- PURCHASE / PROCUREMENT NOTE (Official PSU/Government Format):\n"
        "A Purchase Note is a JUSTIFICATION and DECISION-SUPPORT document for procurement approval.\n"
        "It is NOT a payment form, billing certificate, or transaction record.\n"
        "The purpose is to present a thorough operational case for why the procurement is necessary.\n"
        "\n"
        "Structure MUST follow this exact order:\n"
        "\n"
        "  # Purchase Note: [Item or Service Being Procured]\n"
        "  Note No.: __________________          Date: __________________\n"
        "  Department: __________________\n"
        "  Subject: Purchase of [item/service description]\n"
        "  ---\n"
        "\n"
        "  ## 1. Purpose and Operational Requirement\n"
        "  Write 3-5 numbered paragraphs explaining:\n"
        "  - What exactly is being procured and why it is needed operationally\n"
        "  - What specific operational gap, problem, or business need drives this requirement\n"
        "  - Which department, team, or process will use it and how\n"
        "  - What the current situation is without this procurement\n"
        "  - How it aligns with HPCL's operational or strategic objectives\n"
        "\n"
        "  ## 2. Specification of Items / Services\n"
        "  Provide a markdown table:\n"
        "  | S.No | Item / Service Description | Technical Specification | Quantity | Est. Unit Cost (Rs.) | Total Est. Cost (Rs.) |\n"
        "  |------|---------------------------|------------------------|----------|---------------------|----------------------|\n"
        "  For unknown values, write: Rs. __________________ or Qty: __________________\n"
        "\n"
        "  ## 3. Operational Justification and Risk of Non-Procurement\n"
        "  Write 3-5 numbered paragraphs covering:\n"
        "  - Why these specific technical specifications are required\n"
        "  - What alternatives were considered and why they were ruled out\n"
        "  - What specific operational, safety, compliance, or financial risks arise if NOT procured\n"
        "  - What the consequence to service delivery or production continuity would be\n"
        "\n"
        "  ## 4. Budget Provision and Financial Considerations\n"
        "  Write 2-3 numbered paragraphs covering:\n"
        "  - The budget head under which this expenditure falls\n"
        "  - Whether budget has been sanctioned or is being sought\n"
        "  - The estimated total cost\n"
        "  Use: Budget Head: __________________ or Estimated Cost: Rs. __________________ for unknowns.\n"
        "\n"
        "  ## 5. Procurement Method and Vendor Strategy\n"
        "  Write 2-3 numbered paragraphs covering:\n"
        "  - The recommended procurement method (Open Tender / Limited Tender / Single Source / Rate Contract / GeM)\n"
        "  - The operational rationale for choosing this method\n"
        "  - Any empanelled vendors or prior procurement history\n"
        "  - Compliance with GFR 2017 / HPCL procurement guidelines\n"
        "\n"
        "  ## 6. Proposed Delivery Schedule\n"
        "  Write 1-2 numbered paragraphs covering:\n"
        "  - Expected timeline for delivery, commissioning, or completion\n"
        "  - Any critical dependencies or go-live milestones\n"
        "\n"
        "  ## 7. Recommendation and Approval Sought\n"
        "  Write a clear, formal recommendation paragraph followed by the exact sanction being sought.\n\n"
        "  Prepared by: __________________\n"
        "  Designation: __________________\n"
        "  Department: __________________\n\n"
        "  Recommended by: __________________\n"
        "  Designation: __________________\n"
        "  Department: __________________\n\n"
        "  Approved by (Competent Authority): __________________\n"
        "  Designation: __________________\n"
        "  Date: __________________\n"
        "\n"
        "CRITICAL RULES:\n"
        "- This document MUST read as a professional procurement justification written by a government officer.\n"
        "- Section 1 and Section 3 MUST be substantive (3-5 numbered paragraphs each) -- NEVER one-liners.\n"
        "- Section 2 MUST be a properly formatted markdown table with all 6 columns.\n"
        "- Section 5 MUST name the recommended procurement method with GFR 2017 reference.\n"
        "- Section 7 MUST state the exact rupee amount or Rs. __________________ and end with full signatory chain.\n"
        "- Use underlines (__________________) for unknowns -- NEVER brackets like [TBD] or [Cost].\n"
        "- Use numbered paragraphs (1., 2., 3.) inside every section -- NEVER bullet points.\n"
        "- Do NOT generate [CHART: ...] markers in this document type.\n"
        "- Do NOT copy these instructions into the output. Write actual professional content.\n"
        "- The output must be a COMPLETED, FINALIZED document -- not a skeleton or template.\n"
        "- SUBJECT-SPECIFIC: Operational justifications and technical specs must be specific to the\n"
        "  item/service requested (e.g., processor speed and RAM if laptops; throughput and uptime\n"
        "  SLA if software platform) -- never generic business benefits.\n"
        "- BUDGET: Section 4 must reference a specific budget head (e.g., 'Capital Expenditure --\n"
        "  IT Infrastructure') or use __________________ -- never invent specific rupee figures.\n"
    ),
 
    "office_order": (
        "\nDOCUMENT TYPE -- OFFICE ORDER (Official PSU/Government Format):\n"
        "An Office Order conveys official decisions, transfers, promotions, or directives.\n"
        "Structure MUST follow this exact order:\n"
        "  # Office Order No. __________________\n"
        "  Date: __________________\n"
        "  Subject: [subject]\n"
        "  ---\n"
        "  ## Order\n"
        "    (Open with 'It is ordered that...' or 'Sanction is hereby accorded to...')\n"
        "    (Numbered paragraphs containing the specific directive)\n"
        "  ## Effective Date\n"
        "    (State when this order comes into effect)\n"
        "  ## Compliance\n"
        "    (Who must comply and any reporting requirement)\n\n"
        "  By Order of Competent Authority\n\n"
        "  Signature: __________________\n"
        "  Name: __________________\n"
        "  Designation: __________________\n"
        "  Department: __________________\n\n"
        "  Copy to:\n"
        "  1. All Concerned Officers\n"
        "  2. HR Division / Personal File\n"
        "  3. Office Copy\n"
        "CRITICAL RULES:\n"
        "- Open with 'It is ordered that...' or 'Sanction is hereby accorded to...'\n"
        "- Number all paragraphs inside ## Order sequentially.\n"
        "- The Order section MUST have at least 3 substantive numbered paragraphs.\n"
        "- State effective date and compliance requirements explicitly.\n"
        "- The Compliance section must name the responsible officer/department and reporting frequency.\n"
        "- Do NOT generate [CHART: ...] markers in this document type.\n"
        "- Use underlines (________________) for any field values not provided in context.\n"
    ),
 
    "training_material": (
        "\nDOCUMENT TYPE -- EDUCATIONAL / TRAINING MATERIAL:\n"
        "Generate the document title directly from the source material. Never invent generic titles.\n"
        "Structure the output as a proper training document:\n"
        "  ## Learning Objectives (4-6 measurable objectives)\n"
        "  ## [Topic Sections -- one per major concept]\n"
        "  ## Assessment (5 questions with answers based only on source content)\n"
        "  ## Glossary (key terms with definitions)\n"
        "  ## Key Takeaways\n"
        "Do NOT create fictional HPCL scenarios, case studies, or KPI dashboards.\n"
    ),
 
    "policy_document": (
        "\nDOCUMENT TYPE -- POLICY / COMPLIANCE DOCUMENT:\n"
        "Structure the output as:\n"
        "  ## Purpose & Scope\n"
        "  ## Applicability\n"
        "  ## Key Requirements\n"
        "  ## Roles & Responsibilities\n"
        "  ## Procedures / Implementation Steps\n"
        "  ## Non-Compliance Consequences\n"
        "  ## Compliance Checklist\n"
        "For every requirement: state WHAT must be done, WHO is responsible, WHEN, and consequences.\n"
    ),
 
    "sop": (
        "\nDOCUMENT TYPE -- STANDARD OPERATING PROCEDURE (SOP):\n"
        "Structure the output as:\n"
        "  ## Purpose & Scope\n"
        "  ## Prerequisites\n"
        "  ## Procedure (numbered steps; each: Action -> Role -> Tool -> Safety Note)\n"
        "  ## Critical Control Points\n"
        "  ## Quality Checks\n"
        "  ## Safety Warnings\n"
        "  ## Troubleshooting (symptom -> cause -> corrective action)\n"
        "  ## Performance Metrics\n"
    ),
 
    "business_report": (
        "\nDOCUMENT TYPE -- BUSINESS / ANALYTICAL REPORT:\n"
        "Structure the output as:\n"
        "  ## Executive Summary  (headline insight -> root cause -> 90-day action)\n"
        "  ## KPI Dashboard  (table: Metric | Value | Trend)\n"
        "     Note: If targets are explicitly available in the data, you can use: Metric | Value | Target | Variance | Trend. Otherwise, DO NOT include Target and Variance columns; use only Metric | Value | Trend.\n"
        "  ## Key Findings\n"
        "     For each finding, you MUST follow this exact consultant-grade format:\n"
        "     * **Observation**: [What happened]\n"
        "     * **Evidence**: [The exact numbers, values, and comparisons from the data]\n"
        "     * **Why it matters**: [The operational or business impact]\n"
        "     * **Recommended action**: [The specific, qualitative action to take next]\n"
        "  ## [Topic-specific analysis sections]\n"
        "  ## Risk Assessment  (table: Risk | Likelihood | Impact | Mitigation)\n"
        "  ## Strategic Recommendations\n"
        "     Format: Recommendation | Operational Rationale | Expected Outcome & Metrics\n"
    ),
 
    "financial_report": (
        "\nDOCUMENT TYPE -- FINANCIAL REPORT:\n"
        "Structure the output as:\n"
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
        "Structure the output as:\n"
        "  ## Operational Summary\n"
        "  ## Category-wise Breakdown\n"
        "  ## Anomalies & Flags\n"
        "  ## Operational Recommendations\n"
        "  ## Performance Metrics\n"
    ),
 
    "informational": (
        "\nDOCUMENT TYPE -- INFORMATIONAL DOCUMENT:\n"
        "Structure the output with:\n"
        "  ## Introduction\n"
        "  ## [Topic sections derived from the content]\n"
        "  ## Summary\n"
        "Present information accurately. Do NOT invent analysis, KPIs, or recommendations.\n"
    ),
 
    "educational": (
        "\nDOCUMENT TYPE -- EDUCATIONAL CONTENT:\n"
        "Write a clear educational document covering the topics in the source material.\n"
        "Structure: Introduction -> topic-by-topic sections -> Key Takeaways.\n"
        "Do NOT reframe as a KPI dashboard or HR analytics report.\n"
    ),
 
    "policy": (
        "\nDOCUMENT TYPE -- POLICY DOCUMENT:\n"
        "Write a structured policy document: Purpose & Scope -> Requirements -> Procedures\n"
        "-> Responsibilities -> Compliance. Present faithfully -- no invented analysis.\n"
    ),
}
 
# -- System prompts ------------------------------------------------------------
SYSTEM_PROMPTS = {
 
    "docx": (
        "You are an expert enterprise document generator. Customize branding and terminology "
        "to match the organization and domain of the input files or user request. If the context is "
        "explicitly HPCL (Hindustan Petroleum Corporation Limited), use HPCL branding and downstream "
        "petroleum paradigms; otherwise, remain domain-agnostic and use the terminology of the file's industry. "
        "Generate detailed, professional reports in markdown format.\n\n"
        "FORMATTING:\n"
        "- Use ## for section headings, ### for subsections.\n"
        "- Use bullet points (- ) for lists.\n"
        "- Use markdown tables where structured data fits.\n"
        "- Write for a corporate audience -- formal tone, no casual language.\n"
        "- Minimum 6 sections with meaningful depth per section.\n"
        "- Each section must have at least 3 sentences of real analysis.\n"
    ),
 
    "pdf": (
        "You are an expert enterprise PDF report generator. Customize branding and terminology "
        "to match the organization and domain of the input files or user request. "
        "Generate structured, professional markdown content with clear headings.\n\n"
        "FORMATTING:\n"
        "- Use ## for major sections, ### for subsections.\n"
        "- Write for a corporate audience -- formal tone, no casual language.\n"
        "- Each section must have at least 3 sentences of real analysis.\n"
    ),
 
    "pptx": (
        "You are an expert PowerPoint presentation generator. Customize branding and terminology "
        "to match the organization and domain of the input files or user request. "
        "Generate concise, slide-ready content.\n\n"
        "SLIDE STRUCTURE RULES:\n"
        "1. Use ## for each slide heading -- each ## becomes exactly one slide.\n"
        "2. Write 4 to 6 bullet points per slide using - \n"
        "3. Each bullet must be ONE complete sentence, maximum 20 words.\n"
        "4. NEVER write ## Title Page, ## Acknowledgement, ## Thank You.\n"
        "5. Total ## sections must be between 10 and 14.\n"
        "6. Every bullet must contain a real insight, number, or recommendation.\n"
        "7. Each slide must cover a DISTINCT topic with no substantial overlap.\n"
    ),
 
    "official_doc": (
        "You are an expert in drafting official government and PSU (Public Sector Undertaking) "
        "documents. Customize the organization name in headers, templates, and signatory blocks "
        "to match the domain of the input files or user request. If the context is explicitly HPCL "
        "(Hindustan Petroleum Corporation Limited), draft the document for HPCL.\n\n"
 
        "CORE IDENTITY OF THIS TASK:\n"
        "You are acting as a senior officer drafting an official document. "
        "Your output must read as a real, complete, professionally written government document -- "
        "not a template, not a form, not a skeleton with blanks everywhere.\n\n"
 
        "CONTENT GENERATION RULES -- MANDATORY:\n"
        "1. Generate SUBSTANTIVE CONTENT for every section. Do not produce one-line sections.\n"
        "2. For Purchase Notes: Sections 1 and 3 must each have 3-5 numbered paragraphs of real\n"
        "   operational reasoning based on the context given.\n"
        "3. For File Notes: Background and Analysis sections must have 3+ numbered paragraphs.\n"
        "4. For Circulars and Notices: Instructions must be a numbered list of minimum 6 specific directives.\n"
        "5. Derive content from whatever context is given in the prompt (subject matter, department,\n"
        "   type of item/policy). A request for 'purchase note for laptops' should produce real\n"
        "   operational justification about why laptops are needed -- not blank lines.\n"
        "6. FULL SIGNATORY CHAIN MANDATORY: For all official document types, you MUST fully render\n"
        "   the signature workflow block. Do NOT omit Prepared by, Recommended by, or Approved by.\n"
        "7. SPECIFIC CONTENT: Ensure all content is highly specific to the subject matter,\n"
        "   not generic consulting phrases.\n\n"
  
        "ANTI-HALLUCINATION RULES -- FOR METADATA FIELDS ONLY:\n"
        "1. NEVER write real or fictional person names. Use blank underlines: __________________\n"
        "2. NEVER invent specific reference numbers. Use: Reference No.: __________________\n"
        "3. NEVER write specific dates unless given in the prompt. Use: Date: __________________\n"
        "4. NEVER invent specific financial figures. Use: Rs. __________________ for unknown costs.\n\n"
 
        "ADAPTIVE CONTENT RULES:\n"
        "1. USE PROVIDED INFORMATION FIRST: Use any specific details from the prompt exactly.\n"
        "2. FILL IN LOGICAL CONTENT: For operational sections, generate realistic, professional\n"
        "   organization-appropriate content based on what has been asked for.\n"
        "3. PROFESSIONAL FALLBACKS: Use professional generic language rather than brackets:\n"
        "   - 'standard hardware specifications' instead of '[specs]'\n"
        "   - 'the approved budget allocation' instead of '[budget]'\n\n"
 
        "FORMATTING RULES:\n"
        "- Follow the EXACT format structure provided in the user prompt.\n"
        "- Use formal, impersonal government language throughout.\n"
        "- NEVER use first-person singular ('I'). Use 'the undersigned', 'it is submitted'.\n"
        "- Number all paragraphs within body sections sequentially (1., 2., 3. ...).\n"
        "- Do NOT add sections not specified in the format.\n"
        "- Do NOT generate charts, KPI dashboards, or analytical summaries.\n"
        "- Start your response with a # heading containing the document type and subject.\n\n"
    ),
 
    "docx_noinput": (
        "You are an expert enterprise document writer. "
        "The user wants a document created from scratch with no source data.\n\n"
        "FORMATTING:\n"
        "- The first section MUST be ## Assumptions & Disclaimer.\n"
        "- Use ## for section headings, ### for subsections.\n"
        "- Be specific and realistic -- no generic filler content.\n"
        "- Minimum 6 sections, each with meaningful, domain-appropriate content.\n"
        "- Do NOT generate [CHART: ...] markers -- no data to chart.\n"
    ),
 
    "pdf_noinput": (
        "You are an expert enterprise PDF writer. "
        "The user wants a PDF document from scratch with no source data.\n\n"
        "FORMATTING:\n"
        "- The first section MUST be ## Assumptions & Disclaimer.\n"
        "- Use ## for major sections, ### for subsections.\n"
        "- Be specific and realistic -- no generic filler content.\n"
        "- Minimum 6 sections with meaningful depth.\n"
        "- Do NOT generate [CHART: ...] markers -- no data to chart.\n"
    ),
 
    "pptx_noinput": (
        "You are an expert PowerPoint generator. "
        "The user wants a presentation from scratch with no source data.\n\n"
        "SLIDE RULES:\n"
        "1. Use ## for each slide heading -- 10 to 14 slides total.\n"
        "2. Write 4 to 6 bullets per ## using - \n"
        "3. Each bullet max 20 words, one complete sentence.\n"
        "4. NEVER write ## Title Page, ## Acknowledgement, ## Thank You.\n"
        "5. The FIRST slide must be ## Assumptions & Disclaimer.\n"
        "6. Do NOT generate [CHART: ...] markers.\n"
    ),
 
    "plan": (
        "You are a document title generator. Your only job is to output a single "
        "5-7 word professional document title based on the file content and user request provided.\n\n"
        "Rules:\n"
        "- Output ONLY the title -- no quotes, no punctuation at the end, no preamble.\n"
        "- Derive the title from the ACTUAL FILE CONTENT described.\n"
        "- NEVER produce generic titles like 'HPCL Training Document', 'KPI Dashboard'.\n"
        "- Only include 'HPCL' if the content is specifically about HPCL operations.\n"
        "- Use correct casing. Always write 'HPCL', 'HR', 'FY', 'KPI' in full uppercase.\n"
    ),
}
 
_OFFICIAL_DOC_TYPES = {
    "file_note", "office_memorandum", "office_notice",
    "circular", "purchase_note", "office_order"
}
 
 
def get_system_prompt(output_type: str, doc_type: str = None, has_files: bool = True) -> str:
    if output_type == "plan":
        return SYSTEM_PROMPTS["plan"]
 
    if doc_type in _OFFICIAL_DOC_TYPES:
        return SYSTEM_PROMPTS["official_doc"] + "\n\n" + _BASE_CORE_RULES
 
    if not has_files:
        key = f"{output_type}_noinput"
        prompt = SYSTEM_PROMPTS.get(key, SYSTEM_PROMPTS.get(output_type, SYSTEM_PROMPTS["docx"]))
        return prompt + "\n\n" + _CORE_RULES + "\n\n" + _NOINPUT_DATA_RULES
        
    prompt = SYSTEM_PROMPTS.get(output_type, SYSTEM_PROMPTS["docx"])
    rules  = _CORE_RULES + "\n\n" + _BASE_GROUNDING_RULES
    
    is_analytical = doc_type in ("business_report", "financial_report", "operational", "analytical")
    if is_analytical:
        rules += "\n\n" + _ANALYTICAL_RULES
        
    rules += "\n\n" + _MULTIFILE_RULES
    
    if is_analytical:
        if output_type in ("docx", "pdf"):
            chart_instructions = _CHART_COVERAGE_RULES + "\n" + CHART_MARKER_GUIDE
        elif output_type == "pptx":
            chart_instructions = (
                "\n25. For analytical reports, distribute charts across the deck where meaningful numeric data exists.\n"
                "\n" + CHART_MARKER_GUIDE
            )
        else:
            chart_instructions = ""
        return prompt + "\n\n" + rules + "\n\n" + chart_instructions
        
    return prompt + "\n\n" + rules
 
 
def get_doctype_rules(doc_type: str) -> str:
    return _DOCTYPE_OUTPUT_RULES.get(doc_type, _DOCTYPE_OUTPUT_RULES["informational"])
 
 
def get_cross_file_rules() -> str:
    return _CROSS_FILE_RULES
 
 
def is_official_doc_type(doc_type: str) -> bool:
    return doc_type in _OFFICIAL_DOC_TYPES
 
 
def query_llama(prompt: str, output_type: str = "docx",
                is_combined: bool = False,
                has_files: bool = True,
                system_override: str = None,
                doc_type: str = None) -> str:
    """
    Call the Groq LLaMA model with exponential backoff for rate limits.
    """
    if system_override:
        system_prompt = system_override
    else:
        system_prompt = get_system_prompt(output_type, doc_type, has_files)
        logging.info(
            f"[query_llama] output_type='{output_type}', "
            f"doc_type='{doc_type}', has_files={has_files}"
        )
 
    if doc_type and doc_type in _OFFICIAL_DOC_TYPES:
        max_tokens = 2048
    else:
        max_tokens = 2048 if is_combined else 1536
 
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
 
    models_to_try = [MODEL_NAME, FALLBACK_MODEL]
 
    for model in models_to_try:
        for attempt in range(4):
            try:
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0.4,
                    "max_tokens":  max_tokens,
                }
 
                logging.info(
                    f"Groq call: model={model}, attempt={attempt + 1}, "
                    f"output_type={output_type}, doc_type={doc_type}, "
                    f"max_tokens={max_tokens}"
                )
 
                response = requests.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=90,
                )
 
                # ── 413 Payload Too Large ───────────────────────────────────
                if response.status_code == 413:
                    logging.error(f"Groq API 413 Response content: {response.text}")
                    if max_tokens > 512:
                        new_max = max(512, max_tokens // 2)
                        logging.warning(
                            f"Payload too large (413) on {model}. "
                            f"Reducing max_tokens {max_tokens} -> {new_max}."
                        )
                        max_tokens = new_max
                        time.sleep(2)
                        continue
                    else:
                        logging.error(
                            f"Payload too large (413) on {model} even at "
                            f"min max_tokens. Trying next model."
                        )
                        break

                # ── 429 Rate Limit -- FIXED: exponential backoff ─────────
                if response.status_code == 429:
                    # Try to read Retry-After from headers first
                    retry_after = (
                        response.headers.get("Retry-After")
                        or response.headers.get("x-ratelimit-reset-requests")
                        or response.headers.get("x-ratelimit-reset")
                    )

                    # Try to parse it
                    server_wait = None
                    if retry_after:
                        try:
                            raw = str(retry_after)
                            nums = re.findall(r"\d+\.?\d*", raw)
                            if nums:
                                val = float(nums[0])
                                # Convert ms to seconds if needed
                                if "ms" in raw.lower():
                                    val = val / 1000.0
                                server_wait = val
                        except Exception:
                            pass

                    # Exponential backoff: 15s, 30s, 60s, 90s
                    # Add jitter to avoid thundering herd
                    base_wait = min(15 * (2 ** attempt), 90)
                    jitter    = random.uniform(0, 5)
                    wait      = max(server_wait or 0, base_wait) + jitter

                    # If this is not the last model, and wait time is too long (> 15 seconds), fallback immediately
                    if model != models_to_try[-1] and wait > 15:
                        logging.warning(
                            f"Rate limited (429) on {model}. Wait time {wait:.1f}s is too long. "
                            f"Switching to fallback model immediately."
                        )
                        break

                    # Cap the sleep time to a maximum of 30 seconds to prevent long UI hangs
                    if wait > 30:
                        logging.warning(
                            f"Capping rate limit sleep duration from {wait:.1f}s to 30.0s."
                        )
                        wait = 30.0

                    logging.warning(
                        f"Rate limited (429) on {model}, attempt {attempt + 1}. "
                        f"server_wait={server_wait}, computed_wait={wait:.1f}s. "
                        f"Sleeping..."
                    )
                    time.sleep(wait)
                    continue

                # ── 503 Service Unavailable ──────────────────────────────
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
                logging.warning(
                    f"Timeout on {model} attempt {attempt + 1}. Retrying."
                )
                time.sleep(8)
 
            except requests.exceptions.ConnectionError:
                logging.warning(
                    f"Connection error on {model} attempt {attempt + 1}. Retrying."
                )
                time.sleep(8)
 
            except requests.exceptions.HTTPError as e:
                logging.warning(
                    f"HTTP error on {model} attempt {attempt + 1}: {e}"
                )
                time.sleep(5)
 
            except Exception as e:
                logging.warning(
                    f"Unexpected error on {model} attempt {attempt + 1}: {e}"
                )
                time.sleep(5)
 
        logging.warning(f"All attempts failed for {model}, trying next model.")
 
    raise Exception(
        "Groq API failed after all retries on all models. "
        "Check logs for rate limit or connectivity details."
    )
