 # rev.17: Based on rev.16, but uses po_followup_rev17 table and new column set

# --- BEGIN COPY OF REV.16 WITH UPDATES FOR REV.17 ---

import gradio as gr
import psycopg2
from psycopg2 import Error
import os
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv
import requests
import difflib
from datetime import datetime

# Load environment variables
load_dotenv()

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'  # Updated to public
}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:latest"
OLLAMA_EXTRACTION_MODEL = "mistral:instruct"

# Use the new table for rev.17
NEW_TABLE = "po_followup_rev17"

# Column name mapping for rev.17 (all lowercase, as per table)
COLUMN_MAP = {
    "cost_center_number": "cost_center_number",
    "project_name": "project_name",
    "po_num": "po_num",
    "po_status": "po_status",
    "vendor": "vendor",
    "approved_date": "approved_date",
    "po_comments": "po_comments",
    "description": "description",
    "uom": "uom",
    "unit_price": "unit_price",
    "currency": "currency",
    "amount": "amount",
    "term": "term",
    "qty_delivered": "qty_delivered"
}

# The rest of the code is copied from rev.16, but all references to columns and the table are updated to use the above.
# The LLM/system prompt for SQL generation is also updated to describe the new columns only.

# (The remainder of the code is identical to rev.16, except for the following key changes:)
# - fetch_unique_list, initialize_unique_lists, and all SQL queries use the new table and column names
# - generate_sql_query and LLM prompt describe only the new columns, with clear types and meanings
# - UI and logic otherwise unchanged

# --- END OF HEADER ---

# Conversation history management and all logic below is copied from rev.16, but updated for new table/columns.

class ConversationManager:
    def __init__(self, max_history=5):
        self.conversation_history = []
        self.detected_entities = {}
        self.last_query_result = None
        self.last_sql_query = None
        self.max_history = max_history
        self.memory_was_cleared = False
    def add_interaction(self, question: str, answer: str, entities: Dict, sql_query: str = None, query_result: Dict = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.memory_was_cleared = False
        if entities.get("project"):
            self.detected_entities["project"] = entities["project"]
        if entities.get("supplier"):
            self.detected_entities["supplier"] = entities["supplier"]
        if query_result:
            self.last_query_result = query_result
        if sql_query:
            self.last_sql_query = sql_query
        self.conversation_history.append({
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "entities": {k: v for k, v in entities.items() if k in ["project", "supplier"] and v is not None}
        })
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    def get_context_for_llm(self) -> str:
        if not self.conversation_history:
            return ""
        context = "Previous conversation:\n"
        for i, interaction in enumerate(self.conversation_history):
            context += f"User: {interaction['question']}\n"
            context += f"Assistant: {interaction['answer']}\n"
        return context
    def get_active_entities(self) -> Dict:
        return self.detected_entities
    def reset(self):
        self.conversation_history = []
        self.detected_entities = {}
        self.last_query_result = None
        self.last_sql_query = None
        self.memory_was_cleared = True

CONVERSATION = ConversationManager()

def fetch_unique_list(column: str) -> list:
    connection = None
    try:
        connection = connect_to_database()
        if not connection:
            print(f"Error connecting to database when fetching unique {column}")
            return []
        pg_column = column
        if column.lower() in COLUMN_MAP:
            pg_column = COLUMN_MAP[column.lower()]
        cursor = connection.cursor()
        cursor.execute(f"SELECT DISTINCT {pg_column} FROM {NEW_TABLE} WHERE {pg_column} IS NOT NULL AND {pg_column} != '' ORDER BY {pg_column}")
        results = [row[0] for row in cursor.fetchall() if row[0]]
        return results
    except Exception as e:
        print(f"Error fetching unique {column}: {e}")
        return []
    finally:
        if connection:
            connection.close()

def initialize_unique_lists():
    global UNIQUE_PROJECTS, UNIQUE_SUPPLIERS
    UNIQUE_PROJECTS = fetch_unique_list("project_name")
    UNIQUE_SUPPLIERS = fetch_unique_list("vendor")

def detect_entities_with_llm(question: str, use_history: bool = True) -> dict:
    try:
        system_message = (
            "You are an entity detection system. Your task is to identify project names and vendor names mentioned in questions about purchase orders.\n"
            "Analyze the question and extract ONLY project names or vendor names if they exist.\n"
            "If this is a follow-up question that refers to previously mentioned entities but doesn't explicitly name them, use the entities from the conversation history.\n"
            "Return your answer in JSON format with the following structure:"
            "{\"project\": \"project name or null if none mentioned\", \"supplier\": \"vendor name or null if none mentioned\"}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        if use_history and CONVERSATION.conversation_history and not CONVERSATION.memory_was_cleared:
            context = CONVERSATION.get_context_for_llm()
            active_entities = CONVERSATION.get_active_entities()
            context_message = context
            if active_entities:
                context_message += "\n\nActive entities in conversation:"
                if 'project' in active_entities:
                    context_message += f"\nProject: {active_entities['project']}"
                if 'supplier' in active_entities:
                    context_message += f"\nSupplier: {active_entities['supplier']}"
            messages.append({"role": "system", "content": context_message})
        if CONVERSATION.memory_was_cleared:
            CONVERSATION.memory_was_cleared = False
        messages.append({"role": "user", "content": question})
        payload = {
            "model": OLLAMA_MODEL,
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

def extract_entity_from_question(question: str, entity_type: str, entity_list: list) -> Tuple[Optional[str], float]:
    question_lower = question.lower()
    # 1. Direct substring match anywhere in entity
    for entity in entity_list:
        if question_lower in entity.lower() or entity.lower() in question_lower:
            return entity, 1.0
    # 2. Try fuzzy match on all substrings of the question (for partials like 'ring road')
    best_match = None
    best_score = 0
    for entity in entity_list:
        entity_lower = entity.lower()
        # Check all substrings of the entity name
        for i in range(len(entity_lower)):
            for j in range(i+3, len(entity_lower)+1):  # Only consider substrings of length >=3
                entity_sub = entity_lower[i:j]
                if entity_sub in question_lower:
                    score = len(entity_sub) / len(entity_lower)
                    if score > best_score:
                        best_score = score
                        best_match = entity
    if best_match and best_score > 0.3:
        return best_match, best_score
    # 3. Fuzzy match using difflib as fallback
    matches = difflib.get_close_matches(question_lower, [e.lower() for e in entity_list], n=1, cutoff=0.5)
    if matches:
        for e in entity_list:
            if e.lower() == matches[0]:
                return e, difflib.SequenceMatcher(None, question_lower, matches[0]).ratio()
    return None, 0.0

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
            # Always match to canonical
            project, conf = extract_entity_from_question(project_candidate, "project", UNIQUE_PROJECTS)
            if not project:
                # Heuristic: extract after 'project' or 'for' in question
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
                # Heuristic: extract after 'supplier' or 'vendor' in question
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
        detected_entities["supplier"] = supplier
        detected_entities["supplier_confidence"] = supplier_confidence
    if use_history and (detected_entities["project"] is None or detected_entities["supplier"] is None):
        active_entities = CONVERSATION.get_active_entities()
        conversation_has_history = len(CONVERSATION.conversation_history) > 0
        if conversation_has_history and detected_entities["project"] is None and "project" in active_entities:
            detected_entities["project"] = active_entities["project"]
            detected_entities["project_confidence"] = 0.8
        if conversation_has_history and detected_entities["supplier"] is None and "supplier" in active_entities:
            detected_entities["supplier"] = active_entities["supplier"]
            detected_entities["supplier_confidence"] = 0.8
    sql_query = generate_sql_query(question, detected_entities, use_history)
    try:
        columns, results = execute_query(sql_query)
        query_result = {"columns": columns, "results": results}
        if columns[0] == "Error":
            answer = results[0][0]
        else:
            answer = generate_natural_language_answer(question, sql_query, columns, results, use_history)
        if use_history:
            CONVERSATION.add_interaction(
                question=question,
                answer=answer,
                entities=detected_entities,
                sql_query=sql_query,
                query_result=query_result
            )
        return answer, sql_query, columns, results, detected_entities
    except Exception as e:
        error_message = f"Error executing query: {str(e)}"
        return error_message, f"Error: {str(e)}", ["Error"], [[error_message]], detected_entities

def generate_sql_query(question: str, detected_entities: dict = None, use_history: bool = True) -> str:
    try:
        if detected_entities is None:
            detected_entities = {}
        # LLM prompt for rev.17: Only describe new columns
        system_message = (
            "You are a SQL query generator. Generate PostgreSQL queries based on natural language questions.\n"
            "IMPORTANT: PostgreSQL is case-sensitive. All column names are lowercase and must be used exactly as shown.\n"
            "The table name is 'po_followup_rev17' and it has these columns (all types in parentheses):\n"
            "- cost_center_number (VARCHAR): Cost center\n"
            "- project_name (VARCHAR): Project name\n"
            "- po_num (VARCHAR): Purchase order number\n"
            "- po_status (VARCHAR): PO status\n"
            "- vendor (VARCHAR): Supplier/vendor name\n"
            "- approved_date (DATE): Date the PO was approved\n"
            "- po_comments (TEXT): Comments about the PO\n"
            "- description (TEXT): Description of the item or PO\n"
            "- uom (VARCHAR): Unit of measure\n"
            "- unit_price (DECIMAL): Unit price for the item\n"
            "- currency (VARCHAR): Currency\n"
            "- amount (DECIMAL): Total amount\n"
            "- term (TEXT): PO terms\n"
            "- qty_delivered (DECIMAL): Quantity delivered\n"
            "You can answer questions about purchase orders, terms, suppliers, projects, items, etc.\n"
            "Only use these columns in your query. Keep the query simple and focused on answering the question. "
            "When matching project_name or vendor by substring, ALWAYS use ILIKE for case-insensitive search (e.g. WHERE project_name ILIKE '%substring%').\n"
            "Return ONLY the raw SQL query, with NO code formatting, NO markdown backticks, and NO explanations.\n"
            "If this is a follow-up question to a previous conversation, consider the context."
        )
        messages = [{"role": "system", "content": system_message}]
        if use_history and CONVERSATION.conversation_history:
            context_message = "Previous conversation context:\n"
            for interaction in CONVERSATION.conversation_history[-2:]:
                context_message += f"User asked: {interaction['question']}\n"
                if CONVERSATION.last_sql_query:
                    context_message += f"SQL used: {CONVERSATION.last_sql_query}\n"
            active_entities = CONVERSATION.get_active_entities()
            if active_entities:
                context_message += "\nActive entities from conversation:\n"
                for entity_type, entity_value in active_entities.items():
                    context_message += f"{entity_type}: {entity_value}\n"
            messages.append({"role": "system", "content": context_message})
        if detected_entities:
            entity_info = "\nEntities in current question:\n"
            if detected_entities.get("project"):
                entity_info += f"project_name: {detected_entities['project']}\n"
            if detected_entities.get("supplier"):
                entity_info += f"vendor: {detected_entities['supplier']}\n"
            if entity_info != "\nEntities in current question:\n":
                messages.append({"role": "system", "content": entity_info})
        messages.append({"role": "user", "content": f"Generate a PostgreSQL query to answer this question: {question}"})
        payload = {
            "model": "gemma3:latest",
            "messages": messages,
            "stream": False
        }
        resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        response = resp.json()
        sql_query = response['choices'][0]['message']['content'].strip()
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        if sql_query.startswith('`') and sql_query.endswith('`'):
            sql_query = sql_query[1:-1].strip()
        return sql_query
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

def generate_natural_language_answer(question: str, sql_query: str, columns: list, results: list, use_history: bool = True) -> str:
    if not results or len(results) == 0:
        return f"There are no results for this query."
    if columns[0] == "Error":
        result_str = results[0][0] if results and len(results) > 0 else "Unknown error"
    else:
        header = " | ".join(columns)
        sep = " | ".join(["---"] * len(columns))
        rows = [" | ".join(str(cell) for cell in row) for row in results]
        result_str = f"{header}\n{sep}\n" + "\n".join(rows)
    prompt = "Given the user's question and the SQL query result below, write a clear, concise answer in English. If the result is empty, say so.\n\n"
    prompt += f"User question: {question}\n\n"
    prompt += f"SQL query: {sql_query}\n\n"
    prompt += f"Query result:\n{result_str}\n\n"
    prompt += "Answer in clear, simple English. Be conversational. Don't repeat all values unless requested."
    if use_history and CONVERSATION.conversation_history:
        prompt += "\n\nThis is part of an ongoing conversation. Previous context:\n"
        for interaction in CONVERSATION.conversation_history[-2:]:
            prompt += f"User: {interaction['question']}\n"
            prompt += f"Assistant: {interaction['answer']}\n"
        prompt += "\nIf this is a follow-up question, refer to entities mentioned in previous questions appropriately. For example, if a project was mentioned before but not in this question, still reference it by name."
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Generate a natural language answer based on the query results."}
    ]
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        response = resp.json()
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error generating natural language answer: {str(e)}"

def connect_to_database():
    try:
        connection = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        cursor = connection.cursor()
        cursor.execute(f"SET search_path TO {DB_CONFIG['schema']}")
        connection.commit()
        cursor.close()
        return connection
    except Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def execute_query(query: str) -> Tuple[List[str], List[List]]:
    connection = None
    try:
        connection = connect_to_database()
        if connection is None:
            return ["Error"], [["Could not connect to database"]]
        cursor = connection.cursor()
        cursor.execute(query)
        if query.strip().upper().startswith("SELECT"):
            columns = [desc[0] for desc in cursor.description]
            results = [list(row) for row in cursor.fetchall()]
        else:
            connection.commit()
            columns = ["Status"]
            results = [[f"Query executed successfully. Affected rows: {cursor.rowcount}"]]
        cursor.close()
        connection.close()
        return columns, results
    except Exception as e:
        if connection:
            connection.close()
        return ["Error"], [[str(e)]]
    finally:
        if connection:
            connection.close()

def create_interface():
    with gr.Blocks(title="RME PO Query Assistant rev.17 (PostgreSQL with Memory)") as interface:
        gr.Markdown("# PO Follow-Up Query AI (rev.17, PostgreSQL with Memory)")
        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                with gr.Row():
                    with gr.Column(scale=8):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about purchase orders, terms, items, suppliers, projects, etc...",
                            lines=2
                        )
                    with gr.Column(scale=2):
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_history_btn = gr.Button("Clear Memory", variant="secondary")
            with gr.Column(scale=3):
                detected_entities = gr.Markdown("### Detected Entities\nNo entities detected yet.", label="Current Entities")
                with gr.Row():
                    supplier_btn = gr.Button("Show Suppliers", elem_id="supplier-btn")
                    project_btn = gr.Button("Show Projects", elem_id="project-btn")
        with gr.Accordion("SQL Query and Results", open=False):
            query_output = gr.Code(label="Current SQL Query", language="sql")
            results = gr.Dataframe(label="Query Results")
        gr.Markdown(
            "### System Information\n"
            "This AI assistant queries the new PO table with selected columns only.\n\n"
            "- Using local PostgreSQL database with the new table: po_followup_rev17\n"
            "- Ask questions about suppliers, terms, projects, etc.\n"
            "- Ask follow-up questions without repeating entities mentioned before\n"
            "- Navigation entity memory is applied to all follow-up questions\n"
        )
        def clear_history():
            global CONVERSATION
            CONVERSATION = ConversationManager()
            return [], "### Detected Entities\nMemory has been reset. The chatbot will no longer remember previous questions and entities.", "", None
        def on_submit(question, chat_history):
            if not question.strip():
                return chat_history, question, "### Detected Entities\nNo query entered", "", None
            try:
                answer, query, columns, results_data, entities = process_question(question, use_history=True)
                chat_history = chat_history + [(question, answer)]
                entities_markdown = "### Detected Entities\n"
                if entities["project"]:
                    entities_markdown += f"**Project:** {entities['project']} *(confidence: {entities['project_confidence']:.2f})\n"
                    if entities.get("project_alternatives") and len(entities["project_alternatives"]) > 0:
                        entities_markdown += "**Alternative projects:** "
                        entities_markdown += ", ".join([f"`{alt}`" for alt in entities["project_alternatives"]])
                        entities_markdown += "\n"
                else:
                    entities_markdown += "**Project:** None\n"
                if entities["supplier"]:
                    entities_markdown += f"**Supplier:** {entities['supplier']} *(confidence: {entities['supplier_confidence']:.2f})\n"
                    if entities.get("supplier_alternatives") and len(entities["supplier_alternatives"]) > 0:
                        entities_markdown += "**Alternative suppliers:** "
                        entities_markdown += ", ".join([f"`{alt}`" for alt in entities["supplier_alternatives"]])
                        entities_markdown += "\n"
                else:
                    entities_markdown += "**Supplier:** None\n"
                if query.startswith("Error") or columns[0] == "Error":
                    error_message = results_data[0][0] if columns[0] == "Error" else query
                    return chat_history, "", entities_markdown, query, gr.Dataframe(value=[[error_message]], headers=["Error"])
                df = gr.Dataframe(value=results_data, headers=columns)
                return chat_history, "", entities_markdown, query, df
            except Exception as e:
                error_message = f"Unexpected error: {str(e)}"
                chat_history = chat_history + [(question, error_message)]
                return chat_history, "", "### Detected Entities\nError in entity detection", "", None
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, detected_entities, query_output, results]
        )
        question_input.submit(
            fn=on_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, detected_entities, query_output, results]
        )
        clear_history_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot, detected_entities, query_output, results]
        )
        supplier_btn.click(None, None, None, js="window.open('/suppliers', '_blank')")
        project_btn.click(None, None, None, js="window.open('/projects', '_blank')")
        gr.Examples(
            examples=[
                "Show me all items for project Rabigh 2 (Mourjan)",
                "What are the terms for supplier الشركة المصرية",
                "List all POs for project MOC HQ at Diriyah-K0005",
                "Show all purchase orders with their terms",
                "Which item has the highest unit price in Ring Road project?",
                "How many items were received for that project?",
                "What were the terms for the last PO you showed me?",
            ],
            inputs=question_input
        )
    return interface

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import webbrowser
app = FastAPI()
def get_unique_list(column: str):
    return fetch_unique_list(column)
@app.get("/suppliers", response_class=HTMLResponse)
def suppliers():
    names = get_unique_list("vendor")
    html = "<h2>Unique Vendor Names</h2><ul>" + "".join(f"<li>{n}</li>" for n in names) + "</ul>"
    return HTMLResponse(content=html)
@app.get("/projects", response_class=HTMLResponse)
def projects():
    names = get_unique_list("project_name")
    html = "<h2>Unique Project Names</h2><ul>" + "".join(f"<li>{n}</li>" for n in names) + "</ul>"
    return HTMLResponse(content=html)
if __name__ == "__main__":
    initialize_unique_lists()
    interface = create_interface()
    app = gr.mount_gradio_app(app, interface, path="/")
    webbrowser.open("http://localhost:7870")
    uvicorn.run(app, host="0.0.0.0", port=7870)
