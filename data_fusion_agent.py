import pandas as pd
import json
import re
import requests
from itertools import combinations
from pathlib import Path

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

def call_llm(prompt: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1500
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=body)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def load_file(path: str) -> pd.DataFrame | str:
    """
    Load any supported file into a DataFrame or raw text.
    Supports: CSV, Excel (.xlsx/.xls), JSON, TSV, plain text.
    Returns a DataFrame for structured files, raw string for unstructured.
    """
    ext = Path(path).suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".json":
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find a list value inside the dict
                for v in data.values():
                    if isinstance(v, list):
                        return pd.DataFrame(v)
            return pd.DataFrame([data])
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        elif ext in (".txt", ".md", ".log"):
            with open(path) as f:
                return f.read()  # raw text, handled separately
        else:
            # Attempt CSV as fallback
            return pd.read_csv(path)
    except Exception as e:
        return f"[ERROR loading {path}: {e}]"


def get_schema_summary(label: str, data, api_key: str) -> dict:
    """
    Use LLM to understand what this file/data is about
    and what each column semantically represents.
    Returns a schema dict: {column_name: semantic_meaning}
    """
    if isinstance(data, pd.DataFrame):
        headers = list(data.columns)
        sample = data.head(5).to_string(index=False)
        prompt = f"""You are a data analyst. Below is a dataset labeled '{label}'.

Columns: {headers}
Sample rows:
{sample}

Task:
1. In one sentence, describe what this dataset is about.
2. For each column, provide its semantic meaning (what real-world concept it represents).

Respond ONLY in this JSON format:
{{
  "description": "...",
  "columns": {{
    "column_name": "semantic meaning",
    ...
  }}
}}"""
    else:
        # Unstructured text file
        preview = str(data)[:1000]
        prompt = f"""You are a data analyst. Below is a text file labeled '{label}'.

Content preview:
{preview}

Task:
1. In one sentence, describe what this file is about.
2. List the key entities or topics mentioned (e.g., department names, employee IDs, product names).

Respond ONLY in this JSON format:
{{
  "description": "...",
  "columns": {{
    "entity_type": "description of key entities found"
  }}
}}"""

    raw = call_llm(prompt, api_key)
    try:
        raw_clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw_clean)
    except Exception:
        return {"description": raw, "columns": {}}


def find_semantic_links(schemas: dict, api_key: str) -> list[dict]:
    """
    Given schemas of all files, ask LLM to find which columns
    across files represent the same real-world entity.
    Returns list of links: [{file_a, col_a, file_b, col_b, entity}]
    """
    schema_text = ""
    for label, schema in schemas.items():
        schema_text += f"\nFile '{label}': {schema['description']}\n"
        for col, meaning in schema.get("columns", {}).items():
            schema_text += f"  - {col}: {meaning}\n"

    prompt = f"""You are a data integration expert. Below are schemas of multiple uploaded files:

{schema_text}

Task:
Identify ALL pairs of columns across different files that represent the same real-world entity or concept (e.g., both refer to 'Department', 'Employee ID', 'Product Code', 'Region', 'Date', etc.).

These are the columns that can be used to JOIN or CORRELATE data across files.

Respond ONLY in this JSON format:
{{
  "links": [
    {{
      "file_a": "label of first file",
      "col_a": "column name in file_a",
      "file_b": "label of second file", 
      "col_b": "column name in file_b",
      "entity": "what real-world concept this represents"
    }}
  ],
  "reasoning": "brief explanation of why these links make sense"
}}

If no meaningful links exist, return {{"links": [], "reasoning": "..."}}"""

    raw = call_llm(prompt, api_key)
    try:
        raw_clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw_clean)
    except Exception:
        return {"links": [], "reasoning": raw}


def fuse_on_links(dataframes: dict, links: list) -> list[dict]:
    """
    Merge DataFrames based on LLM-discovered semantic links.
    Returns list of per-entity insight dicts.
    """
    insights = []

    for link in links:
        fa, ca = link["file_a"], link["col_a"]
        fb, cb = link["file_b"], link["col_b"]
        entity = link["entity"]

        df_a = dataframes.get(fa)
        df_b = dataframes.get(fb)

        if not isinstance(df_a, pd.DataFrame) or not isinstance(df_b, pd.DataFrame):
            continue
        if ca not in df_a.columns or cb not in df_b.columns:
            continue

        try:
            # Normalize for join
            temp_a = df_a.copy()
            temp_b = df_b.copy()
            temp_a["__join_key__"] = temp_a[ca].astype(str).str.strip().str.lower()
            temp_b["__join_key__"] = temp_b[cb].astype(str).str.strip().str.lower()

            merged = pd.merge(temp_a, temp_b, on="__join_key__",
                              suffixes=(f"_{fa}", f"_{fb}"), how="inner")
            merged.drop(columns=["__join_key__"], inplace=True)

            if merged.empty:
                continue

            # Build per-entity-value summaries
            join_col = f"{ca}_{fa}" if ca != cb else ca
            group_col = join_col if join_col in merged.columns else merged.columns[0]

            for val, group in merged.groupby(group_col):
                # Aggregate numeric columns
                numeric_summary = {}
                for col in group.select_dtypes(include='number').columns:
                    numeric_summary[col] = {
                        "sum": round(group[col].sum(), 2),
                        "mean": round(group[col].mean(), 2),
                        "min": round(group[col].min(), 2),
                        "max": round(group[col].max(), 2)
                    }

                insights.append({
                    "entity": entity,
                    "value": val,
                    "sources": [fa, fb],
                    "row_count": len(group),
                    "numeric_summary": numeric_summary,
                    "sample_rows": group.head(3).to_dict(orient="records")
                })

        except Exception as e:
            print(f"[DataFusionAgent] Merge failed for {fa}.{ca} <-> {fb}.{cb}: {e}")

    return insights


def generate_cross_file_insights(fused_data: list, schemas: dict, api_key: str) -> str:
    """
    Ask LLM to generate high-level strategic insights from the fused data.
    This is what gets passed to your document agents.
    """
    if not fused_data:
        return ""

    fused_text = json.dumps(fused_data[:20], indent=2)  # cap to avoid token overflow
    schema_descs = {k: v["description"] for k, v in schemas.items()}

    prompt = f"""You are a senior business analyst. Multiple datasets have been merged:

Files analyzed: {json.dumps(schema_descs, indent=2)}

Merged cross-file data:
{fused_text}

Task:
Generate strategic, analytical insights that CONNECT information across these files.
Do NOT just list numbers. Identify patterns, correlations, anomalies, and actionable findings.

Format your response as:
1. KEY FINDINGS (3-5 bullet points connecting data across files)
2. NOTABLE PATTERNS (trends or correlations discovered)
3. RECOMMENDED FOCUS AREAS (what a decision-maker should act on)"""

    return call_llm(prompt, api_key)


class DataFusionAgent:
    """
    Multi-file data fusion agent that:
    - Accepts any file type (CSV, Excel, JSON, text)
    - Uses LLM to understand each file's schema semantically
    - Discovers cross-file entity links automatically
    - Merges data and generates strategic insights
    - Outputs enriched context ready for document generation agents
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.raw_data = {}       # {label: DataFrame or str}
        self.schemas = {}        # {label: schema dict}
        self.links = []          # discovered semantic links
        self.fused = []          # merged insight dicts
        self.insight_text = ""   # final LLM narrative

    def load(self, files: dict):
        """files: {label: file_path}"""
        for label, path in files.items():
            self.raw_data[label] = load_file(path)
            print(f"[DataFusionAgent] Loaded '{label}' from {path}")

    def analyze(self):
        """Run full pipeline: schema → links → fusion → insights"""

        # Step 1: Schema understanding
        print("[DataFusionAgent] Analyzing schemas...")
        for label, data in self.raw_data.items():
            self.schemas[label] = get_schema_summary(label, data, self.api_key)
            print(f"  '{label}': {self.schemas[label]['description']}")

        # Step 2: Discover semantic links
        print("[DataFusionAgent] Discovering cross-file entity links...")
        link_result = find_semantic_links(self.schemas, self.api_key)
        self.links = link_result.get("links", [])
        print(f"  Found {len(self.links)} link(s). Reasoning: {link_result.get('reasoning', '')}")

        # Step 3: Fuse data
        if self.links:
            print("[DataFusionAgent] Fusing datasets...")
            self.fused = fuse_on_links(self.raw_data, self.links)
            print(f"  Generated {len(self.fused)} cross-file entity records.")

        # Step 4: Generate insights narrative
        print("[DataFusionAgent] Generating strategic insights...")
        self.insight_text = generate_cross_file_insights(self.fused, self.schemas, self.api_key)

    def to_llm_context(self) -> str:
        """
        Returns the full enriched context string to prepend
        to your existing LLM agent prompts.
        """
        parts = []

        parts.append("=== MULTI-FILE ANALYSIS CONTEXT ===\n")

        for label, schema in self.schemas.items():
            parts.append(f"[{label}]: {schema['description']}")

        if self.links:
            parts.append(f"\nDiscovered {len(self.links)} cross-file connection(s):")
            for link in self.links:
                parts.append(f"  • {link['file_a']}.{link['col_a']} ↔ {link['file_b']}.{link['col_b']} (shared entity: {link['entity']})")

        if self.insight_text:
            parts.append(f"\n=== CROSS-FILE INSIGHTS ===\n{self.insight_text}")

        return "\n".join(parts)