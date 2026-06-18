import json
import logging
from llm_agent import query_llama
 
logger = logging.getLogger(__name__)
 
PLANNER_PROMPT = """
You are a document planning agent. Given a user request and a list of uploaded files, 
your job is to break the task into clear subtasks and assign each to the right specialist agent.
 
Available agents:
- data_agent: reads and extracts structured data from a file (CSV, Excel, DOCX, PDF)
- analysis_agent: performs analysis on extracted data (trends, KPIs, comparisons, growth)
- graph_agent: generates charts from real data (bar, line, pie, horizontal_bar)
- insight_agent: generates recommendations, conclusions, cross-file comparisons
- summary_agent: summarizes content from a file
 
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
 
def plan_tasks(prompt, file_names):
    user_msg = f"User request: {prompt}\n\nUploaded files: {', '.join(file_names)}"
    raw = query_llama(user_msg, output_type="plan", system_override=PLANNER_PROMPT)
 
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        plan  = json.loads(raw[start:end])
        logger.info(f"Plan created: {len(plan['subtasks'])} subtasks")
        return plan
    except Exception as e:
        logger.error(f"Planner failed to parse JSON: {e}\nRaw: {raw}")
        return _fallback_plan(prompt, file_names)
 
 
def _fallback_plan(prompt, file_names):
    subtasks = []
    tid = 1
    for f in file_names:
        subtasks.append({"task_id": f"t{tid}", "agent": "data_agent",
                         "file": f, "instruction": f"Extract all data from {f}"})
        subtasks.append({"task_id": f"t{tid+1}", "agent": "analysis_agent",
                         "depends_on": [f"t{tid}"],
                         "instruction": f"Analyze data from {f} and find key insights"})
        subtasks.append({"task_id": f"t{tid+2}", "agent": "graph_agent",
                         "depends_on": [f"t{tid}"],
                         "instruction": f"Generate relevant charts from {f}"})
        tid += 3
 
    subtasks.append({"task_id": f"t{tid}", "agent": "insight_agent",
                     "depends_on": [t["task_id"] for t in subtasks],
                     "instruction": prompt})
 
    return {"document_title": "Analysis Report", "subtasks": subtasks}