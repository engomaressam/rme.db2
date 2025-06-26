import gradio as gr
import psycopg2
from psycopg2 import Error
import os
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dotenv import load_dotenv
import requests
import difflib
from datetime import datetime
import time
import traceback
import hashlib

# Load environment variables
load_dotenv()

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:latest"
OLLAMA_EXTRACTION_MODEL = "mistral:instruct"

# --- Memory and Conversation Management ---
class ConversationMemory:
    def __init__(self):
        self.history = []
        self.active_entities = {"project": None, "supplier": None}
        self.memory_was_cleared = False

    def add_turn(self, user, bot, detected_entities):
        self.history.append({
            "user": user,
            "bot": bot,
            "entities": detected_entities.copy()
        })
        # Update active entities if detected
        for k in ["project", "supplier"]:
            if detected_entities.get(k):
                self.active_entities[k] = detected_entities[k]

    def get_active_entity(self, entity_type):
        return self.active_entities.get(entity_type)

    def clear(self):
        self.history.clear()
        self.active_entities = {"project": None, "supplier": None}
        self.memory_was_cleared = True

CONVERSATION = ConversationMemory()

# --- Unique List Fetching ---
def fetch_unique_list(column):
    try:
        connection = psycopg2.connect(**{k: v for k, v in DB_CONFIG.items() if k != 'schema'})
        cursor = connection.cursor()
        cursor.execute(f"SELECT DISTINCT {column} FROM po_followup_rev17")
        rows = cursor.fetchall()
        return sorted([row[0] for row in rows if row[0]])
    except Exception as e:
        print(f"Error fetching unique list for {column}: {e}")
        return []
    finally:
        if connection:
            connection.close()

UNIQUE_PROJECTS = fetch_unique_list("project_name")
UNIQUE_SUPPLIERS = fetch_unique_list("vendor")

# --- Entity Extraction and Matching ---
def detect_entities_with_llm(question: str, use_history: bool = True) -> dict:
    try:
        # Improved prompt: explicit instructions, examples, and column explanation
        system_prompt = '''
You are an entity detection system. Your task is to identify project names and supplier names mentioned in questions about purchase orders.
The table has these columns (with examples):
- project_name: The name of the project (e.g., 'Ring Road', 'NEOM Smart City')
- vendor: The supplier or vendor name (e.g., 'Siemens', 'المصرية للدرابزين')
- po_num: Purchase order number (e.g., 'PO12345')
- po_status: PO status (e.g., 'Open', 'Closed')
- approved_date: Date the PO was approved (e.g., '2024-05-01')
- po_comments: Comments about the PO
- description: Description of the item or PO
- uom: Unit of measure (e.g., 'kg', 'pcs')
- unit_price: Unit price for the item
- currency: Currency (e.g., 'SAR', 'USD')
- amount: Total amount
- term: PO terms
- qty_delivered: Quantity delivered
Analyze the question and extract ONLY project names or supplier names if they exist.
If this is a follow-up question that refers to previously mentioned entities but doesn't explicitly name them, use the entities from the conversation history.
Return your answer in JSON format with the following structure:
{"project": "project name or null if none mentioned", "supplier": "supplier name or null if none mentioned"}
If a project or supplier is not mentioned in this question or previous context, use null (not empty string).
Example 1: 'Show me all POs for Ring Road project'
Response: {"project": "Ring Road", "supplier": null}
Example 2: 'What are the terms for supplier Siemens?'
Response: {"project": null, "supplier": "Siemens"}
Example 3: [After talking about Ring Road project] 'What items were purchased for it?'
Response: {"project": "Ring Road", "supplier": null}
'''

        messages = [{"role": "system", "content": system_prompt}]
        if use_history and CONVERSATION.history and not CONVERSATION.memory_was_cleared:
            context = f"Previous context: {CONVERSATION.history[-1]['user']} => {CONVERSATION.history[-1]['entities']}"
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": question})
        payload = {
            "model": "gemma3:latest",
            "messages": messages,
            "stream": False
        }
        resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        response = resp.json()
        llm_response = response['choices'][0]['message']['content'].strip()
        json_match = re.search(r'\{[^{}]*\}', llm_response)
        if json_match:
            json_str = json_match.group(0)
            print(f"[LLM ENTITY DEBUG] Raw LLM output: {llm_response}")
            try:
                entities = json.loads(json_str)
                result = {
                    "project": entities.get("project") if entities.get("project") != "null" else None,
                    "supplier": entities.get("supplier") if entities.get("supplier") != "null" else None
                }
                return result
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from LLM response: {json_str}")
        print(f"[LLM ENTITY DEBUG] No JSON match or decode error. Raw output: {llm_response}")
        return {"project": None, "supplier": None}
    except Exception as e:
        print(f"Error in LLM entity detection: {str(e)}")
        return {"project": None, "supplier": None}

def extract_entity_from_question(candidate: str, entity_type: str, entity_list: list) -> Tuple[Optional[str], float]:
    candidate_lower = candidate.lower()
    # 1. Direct substring match
    for entity in entity_list:
        if candidate_lower in entity.lower() or entity.lower() in candidate_lower:
            return entity, 1.0
    # 2. Fuzzy match
    best_match = None
    best_score = 0
    for entity in entity_list:
        score = difflib.SequenceMatcher(None, candidate_lower, entity.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = entity
    if best_match and best_score > 0.5:
        return best_match, best_score
    return None, 0.0

# --- Main Question Processing ---
def process_question(question: str, use_history: bool = True) -> Tuple[str, str, List[str], List[List], Dict]:
    detected_entities = {
        "project": None,
        "project_confidence": 0.0,
        "project_alternatives": [],
        "supplier": None,
        "supplier_confidence": 0.0,
        "supplier_alternatives": []
    }
    llm_entities = detect_entities_with_llm(question, use_history=use_history)
    import re
    if llm_entities is not None:
        # PROJECT extraction
        project_candidate = llm_entities.get("project")
        if project_candidate:
            project, conf = extract_entity_from_question(project_candidate, "project", UNIQUE_PROJECTS)
            if not project:
                match = re.search(r'project\s+([\w\s-]+)', question, re.IGNORECASE)
                if match:
                    heuristic_candidate = match.group(1).strip()
                    project, conf = extract_entity_from_question(heuristic_candidate, "project", UNIQUE_PROJECTS)
            if project:
                detected_entities["project"] = project
                detected_entities["project_confidence"] = conf
                print(f"[ENTITY MATCH] Canonical project used: {project}")
                other_matches = difflib.get_close_matches(project, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                alternatives = [m for m in other_matches if m != project][:2]
                detected_entities["project_alternatives"] = alternatives
        # SUPPLIER extraction
        supplier_candidate = llm_entities.get("supplier")
        if supplier_candidate:
            supplier, conf = extract_entity_from_question(supplier_candidate, "supplier", UNIQUE_SUPPLIERS)
            if not supplier:
                match = re.search(r'(supplier|vendor)\s+([\w\s-]+)', question, re.IGNORECASE)
                if match:
                    heuristic_candidate = match.group(2).strip()
                    supplier, conf = extract_entity_from_question(heuristic_candidate, "supplier", UNIQUE_SUPPLIERS)
            if supplier:
                detected_entities["supplier"] = supplier
                detected_entities["supplier_confidence"] = conf
                print(f"[ENTITY MATCH] Canonical supplier used: {supplier}")
                other_matches = difflib.get_close_matches(supplier, UNIQUE_SUPPLIERS, n=3, cutoff=0.5)
                alternatives = [m for m in other_matches if m != supplier][:2]
                detected_entities["supplier_alternatives"] = alternatives
    # If LLM returned nothing, try regex/heuristic fallback directly on the question
    if not llm_entities or (not llm_entities.get("project") and not llm_entities.get("supplier")):
        project, project_confidence = extract_entity_from_question(question, "project", UNIQUE_PROJECTS)
        if not project:
            match = re.search(r'project\s+([\w\s-]+)', question, re.IGNORECASE)
            if match:
                heuristic_candidate = match.group(1).strip()
                project, project_confidence = extract_entity_from_question(heuristic_candidate, "project", UNIQUE_PROJECTS)
        if project:
            detected_entities["project"] = project
            detected_entities["project_confidence"] = project_confidence
            print(f"[ENTITY MATCH] Canonical project used (fallback): {project}")
        supplier, supplier_confidence = extract_entity_from_question(question, "supplier", UNIQUE_SUPPLIERS)
        if not supplier:
            match = re.search(r'(supplier|vendor)\s+([\w\s-]+)', question, re.IGNORECASE)
            if match:
                heuristic_candidate = match.group(2).strip()
                supplier, supplier_confidence = extract_entity_from_question(heuristic_candidate, "supplier", UNIQUE_SUPPLIERS)
        if supplier:
            detected_entities["supplier"] = supplier
            detected_entities["supplier_confidence"] = supplier_confidence
            print(f"[ENTITY MATCH] Canonical supplier used (fallback): {supplier}")
    # --- Entity Inheritance Logic ---
    # If project or supplier is still None or low confidence, inherit from memory if available
    CONFIDENCE_THRESHOLD = 0.7
    for entity_type in ["project", "supplier"]:
        if (not detected_entities[entity_type] or detected_entities[f"{entity_type}_confidence"] < CONFIDENCE_THRESHOLD):
            inherited = CONVERSATION.get_active_entity(entity_type)
            if inherited:
                detected_entities[entity_type] = inherited
                detected_entities[f"{entity_type}_confidence"] = 1.0
                print(f"[ENTITY INHERIT] Using previous {entity_type}: {inherited}")
    # --- SQL Generation ---
    # Use only canonical names for SQL
    sql_filters = []
    if detected_entities["project"]:
        sql_filters.append(f"project_name = '{detected_entities['project']}'")
    if detected_entities["supplier"]:
        sql_filters.append(f"vendor = '{detected_entities['supplier']}'")
    where_clause = f"WHERE {' AND '.join(sql_filters)}" if sql_filters else ""
    sql_query = f"SELECT * FROM po_followup_rev17 {where_clause} LIMIT 10;"
    # --- Execute SQL ---
    results = []
    try:
        connection = psycopg2.connect(**{k: v for k, v in DB_CONFIG.items() if k != 'schema'})
        cursor = connection.cursor()
        cursor.execute(sql_query)
        if cursor.description:  # Check if there are results
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            # Convert results to list of lists for easier handling
            results = [list(row) for row in results]
            # Add column names as first row
            results.insert(0, columns)
    except Exception as e:
        print(f"SQL error: {e}")
        return f"SQL Error: {str(e)}", [], [], [], detected_entities
    finally:
        if connection:
            connection.close()
    # --- Add to Conversation Memory ---
    CONVERSATION.add_turn(question, str(results), detected_entities)
    return sql_query, results, [], results, detected_entities

def format_results(results):
    if not results or len(results) <= 1:  # Only headers or empty
        return "**No results found.**"
    
    # First row is column headers
    columns = results[0]
    data_rows = results[1:]
    
    # Create markdown table
    table = []
    
    # Add header
    table.append("|" + "|".join(columns) + "|")
    table.append("|" + "|".join(["---"] * len(columns)) + "|")
    
    # Add data rows
    for row in data_rows:
        # Convert all values to strings and escape pipes
        row_str = [str(val).replace("|", "\\|").replace("\n", " ") if val is not None else "" for val in row]
        table.append("|" + "|".join(row_str) + "|")
    
    return "\n".join(table)

def respond(question, history=None):
    try:
        # Process the question
        sql, results, _, _, entities = process_question(question)
        
        # Format the response with Markdown
        response = """## Detected Entities
```json
{0}
```

## SQL Query
```sql
{1}
```

## Results
{2}""".format(
            json.dumps(entities, indent=2, ensure_ascii=False),
            sql,
            format_results(results) if results else "*No results found.*"
        )
        
        return response
    except Exception as e:
        error_msg = f"**Error:** {str(e)}"
        print(f"Error in respond: {str(e)}")
        return error_msg

# Create Gradio interface
with gr.Blocks(title="PO Query Assistant") as demo:
    gr.Markdown("# PO Query Assistant")
    gr.Markdown("Ask questions about purchase orders in natural language.")
    
    with gr.Row():
        question = gr.Textbox(label="Your question", placeholder="E.g., Show me POs for Ring Road")
        submit_btn = gr.Button("Ask")
    
    output = gr.Markdown()
    
    # Handle submit button click
    submit_btn.click(
        fn=respond,
        inputs=[question],
        outputs=output
    )
    
    # Handle pressing Enter in the textbox
    question.submit(
        fn=respond,
        inputs=[question],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7861)
