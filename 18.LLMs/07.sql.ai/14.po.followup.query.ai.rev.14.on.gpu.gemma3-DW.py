import gradio as gr
import mysql.connector
from mysql.connector import Error
import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import requests
import difflib

# Load environment variables
load_dotenv()

DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:latest"

NEW_TABLE = "po_followup_merged"

# Fetch and cache unique projects and suppliers at startup
def fetch_unique_list(column: str) -> list:
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        cursor.execute(f"SELECT DISTINCT {column} FROM {NEW_TABLE} WHERE {column} IS NOT NULL AND {column} != '' ORDER BY {column}")
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
    UNIQUE_PROJECTS = fetch_unique_list("PROJECT_NAME")
    UNIQUE_SUPPLIERS = fetch_unique_list("VENDOR_NAME")

# Enhanced entity extraction and matching

def detect_entities_with_llm(question: str) -> dict:
    """
    Use the LLM to detect project and supplier names mentioned in the question.
    Returns a dictionary with detected entities.
    """
    try:
        system_message = "You are an entity detection system. Your task is to identify project names and supplier names mentioned in questions about purchase orders."
        system_message += "\n\nAnalyze the question and extract ONLY project names or supplier names if they exist."
        system_message += "\n\nReturn your answer in JSON format with the following structure:"
        system_message += "\n{\"project\": \"project name or null if none mentioned\", \"supplier\": \"supplier name or null if none mentioned\"}"
        system_message += "\n\nIf a project or supplier is not mentioned, use null (not empty string)."
        system_message += "\n\nExample 1: 'Show me all POs for Ring Road project'"
        system_message += "\nResponse: {\"project\": \"Ring Road\", \"supplier\": null}"
        system_message += "\n\nExample 2: 'What are the terms for supplier Siemens?'"
        system_message += "\nResponse: {\"project\": null, \"supplier\": \"Siemens\"}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
        
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        }
        
        resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        response = resp.json()
        llm_response = response['choices'][0]['message']['content'].strip()
        
        # Extract JSON from the response
        import json
        import re
        
        # Look for JSON pattern in the response
        json_match = re.search(r'\{[^{}]*\}', llm_response)
        if json_match:
            json_str = json_match.group(0)
            try:
                entities = json.loads(json_str)
                result = {
                    "project": entities.get("project") if entities.get("project") != "null" else None,
                    "supplier": entities.get("supplier") if entities.get("supplier") != "null" else None
                }
                return result
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from LLM response: {json_str}")
        
        # Fallback: return empty results
        return {"project": None, "supplier": None}
        
    except Exception as e:
        print(f"Error in LLM entity detection: {str(e)}")
        return {"project": None, "supplier": None}
def extract_entity_from_question(question: str, entity_type: str, entity_list: list) -> Tuple[Optional[str], float]:
    """
    Extract entity from question and return both the entity and the match confidence score
    """
    # First try direct simple matching from question to entity list
    question_lower = question.lower()
    
    # Direct matching - more reliable for exact phrases in the question
    for entity in entity_list:
        if entity.lower() in question_lower:
            # Direct match found
            return entity, 1.0
    
    # Try keyword matching - look for key parts of project/supplier names
    question_words = set(question_lower.replace('(', ' ').replace(')', ' ').replace(',', ' ').replace('?', ' ').split())
    best_match = None
    highest_score = 0
    
    for entity in entity_list:
        entity_lower = entity.lower()
        entity_words = set(entity_lower.split())
        
        # Check if any significant word from the entity appears in the question
        for word in entity_words:
            if len(word) > 3 and word in question_words:  # Only consider significant words
                # Calculate overlap score
                common_words = len(question_words & entity_words)
                total_words = len(entity_words)
                score = common_words / total_words
                
                if score > highest_score:
                    highest_score = score
                    best_match = entity
    
    if best_match and highest_score > 0.3:  # Minimum threshold
        return best_match, highest_score
    
    # As a last resort, use difflib for fuzzy matching
    # Extract potential entity fragments (2-3 word combinations) from the question
    words = question_lower.split()
    potential_fragments = []
    for i in range(len(words)):
        if i < len(words) - 1:
            potential_fragments.append(f"{words[i]} {words[i+1]}")
        if i < len(words) - 2:
            potential_fragments.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    best_score = 0
    best_entity = None
    
    for fragment in potential_fragments:
        matches = difflib.get_close_matches(fragment, [e.lower() for e in entity_list], n=1, cutoff=0.6)
        if matches:
            score = difflib.SequenceMatcher(None, fragment, matches[0]).ratio()
            if score > best_score:
                best_score = score
                # Find the original entity with case preserved
                for e in entity_list:
                    if e.lower() == matches[0]:
                        best_entity = e
                        break
    
    if best_entity and best_score > 0.6:
        return best_entity, best_score
        
    # No match found
    return None, 0.0

def process_question(question: str) -> Tuple[str, str, list, list, dict]:
    """
    Process the question and return answer, query, columns, results, and detected entities
    """
    # Extract detected entities with alternative matches
    detected_entities = {
        "project": None,
        "project_confidence": 0.0,
        "project_alternatives": [],  # Will store up to 2 alternative matches
        "supplier": None,
        "supplier_confidence": 0.0,
        "supplier_alternatives": []   # Will store up to 2 alternative matches
    }
    
    # First, use the LLM to detect potential project and supplier names
    llm_entities = detect_entities_with_llm(question)
    
    # Check if LLM detected a project (explicit values vs missing keys)
    if "project" in llm_entities:
        project_candidate = llm_entities["project"]
        # If project was detected (not None), try to match it with our database
        if project_candidate:
            # Try to find the best match in our database for the LLM-detected project
            if project_candidate in UNIQUE_PROJECTS:
                # Direct match
                detected_entities["project"] = project_candidate
                detected_entities["project_confidence"] = 1.0
                
                # Get alternatives for even direct matches (similar named projects)
                other_matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                alternatives = [m for m in other_matches if m != project_candidate][:2]  # Take up to 2 alternatives
                detected_entities["project_alternatives"] = alternatives
            else:
                # Find closest match using difflib (get top 3 matches)
                matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                if matches:
                    # Primary match
                    detected_entities["project"] = matches[0]
                    detected_entities["project_confidence"] = difflib.SequenceMatcher(None, project_candidate, matches[0]).ratio()
                    
                    # Store alternatives (2nd and 3rd matches if they exist)
                    detected_entities["project_alternatives"] = matches[1:3] if len(matches) > 1 else []
    else:
        # Only fallback to traditional extraction if LLM failed completely
        # (didn't return a valid response with project field)
        project_guess, project_confidence = extract_entity_from_question(question, "project", UNIQUE_PROJECTS)
        if project_guess:
            detected_entities["project"] = project_guess
            detected_entities["project_confidence"] = project_confidence
    
    # Check if LLM detected a supplier (explicit values vs missing keys)
    if "supplier" in llm_entities:
        supplier_candidate = llm_entities["supplier"]
        # If supplier was detected (not None), try to match it with our database
        if supplier_candidate:
            if supplier_candidate in UNIQUE_SUPPLIERS:
                # Direct match
                detected_entities["supplier"] = supplier_candidate
                detected_entities["supplier_confidence"] = 1.0
                
                # Get alternatives for even direct matches (similar named suppliers)
                other_matches = difflib.get_close_matches(supplier_candidate, UNIQUE_SUPPLIERS, n=3, cutoff=0.5)
                alternatives = [m for m in other_matches if m != supplier_candidate][:2]  # Take up to 2 alternatives
                detected_entities["supplier_alternatives"] = alternatives
            else:
                # Find closest match using difflib (get top 3 matches)
                matches = difflib.get_close_matches(supplier_candidate, UNIQUE_SUPPLIERS, n=3, cutoff=0.5)
                if matches:
                    # Primary match
                    detected_entities["supplier"] = matches[0]
                    detected_entities["supplier_confidence"] = difflib.SequenceMatcher(None, supplier_candidate, matches[0]).ratio()
                    
                    # Store alternatives (2nd and 3rd matches if they exist)
                    detected_entities["supplier_alternatives"] = matches[1:3] if len(matches) > 1 else []
    else:
        # Only fallback to traditional extraction if LLM failed completely
        # (didn't return a valid response with supplier field)
        supplier_guess, supplier_confidence = extract_entity_from_question(question, "supplier", UNIQUE_SUPPLIERS)
        if supplier_guess:
            detected_entities["supplier"] = supplier_guess
            detected_entities["supplier_confidence"] = supplier_confidence
    
    # If an entity was found, rewrite the question with the exact name
    rewritten_question = question
    
    # Prepare an augmented question with explicit entity information
    augmented_question = question
    
    project_match = detected_entities["project"]
    supplier_match = detected_entities["supplier"]
    
    if project_match:
        # Explicitly add the project name to the augmented question
        augmented_question = f"{question} (for project: {project_match})"
    
    if supplier_match:
        # Explicitly add the supplier name 
        if project_match:
            augmented_question += f" (for supplier: {supplier_match})"
        else:
            augmented_question = f"{question} (for supplier: {supplier_match})"
    
    # Use the augmented question for SQL generation to ensure entities are properly used
    query = generate_sql_query(augmented_question)
    
    if query.startswith("Error"):
        return "", query, ["Error"], [["Failed to generate SQL query"]], detected_entities
        
    # Execute the query
    columns, results = execute_query(query)
    
    # Generate natural language answer
    answer = generate_natural_language_answer(rewritten_question, query, columns, results)
    
    return answer, query, columns, results, detected_entities

def generate_sql_query(question: str) -> str:
    try:
        # Explain all columns to the LLM
        system_message = "You are a SQL query generator. Generate MySQL queries based on natural language questions.\n\n"
        system_message += "The table name is 'po_followup_merged' and it has these columns:\n"
        system_message += "- id: auto-increment row id (INT)\n"
        system_message += "- PO_NUM: purchase order number (VARCHAR)\n"
        system_message += "- COMMENTS: comments about the PO (TEXT)\n"
        system_message += "- APPROVED_DATE: date the PO was approved (DATE)\n"
        system_message += "- UOM: unit of measure (VARCHAR)\n"
        system_message += "- ITEM_DESCRIPTION: description of the item (TEXT)\n"
        system_message += "- UNIT_PRICE: unit price for the item (DECIMAL)\n"
        system_message += "- QUANTITY_RECEIVED: quantity received (DECIMAL)\n"
        system_message += "- LINE_AMOUNT: total line amount (DECIMAL)\n"
        system_message += "- PROJECT_NAME: project name (VARCHAR)\n"
        system_message += "- VENDOR_NAME: supplier/vendor name (VARCHAR)\n"
        system_message += "- TERMS: merged PO terms (TEXT)\n\n"
        system_message += "You can answer questions about purchase orders, terms, suppliers, projects, items, etc.\n\n"
        system_message += "Only use these columns in your query. Keep the query simple and focused on answering the question. "
        system_message += "Return ONLY the raw SQL query, with NO code formatting, NO markdown backticks, NO ```sql tags, and NO explanations."
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Generate a MySQL query to answer this question: {question}"}
        ]
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False
        }
        
        resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        response = resp.json()
        sql_query = response['choices'][0]['message']['content'].strip()
        
        # Clean the SQL query - Remove any markdown code fences or backticks
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        if sql_query.startswith('`') and sql_query.endswith('`'):
            sql_query = sql_query[1:-1].strip()
            
        return sql_query
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

def generate_natural_language_answer(question: str, sql_query: str, columns: list, results: list) -> str:
    if not results or len(results) == 0:
        return f"There are no results for this query."
        
    # Format results as readable string
    if columns[0] == "Error":
        result_str = results[0][0] if results and len(results) > 0 else "Unknown error"
    else:
        header = " | ".join(columns)
        sep = " | ".join(["---"] * len(columns))
        rows = [" | ".join(str(cell) for cell in row) for row in results]
        result_str = f"{header}\n{sep}\n" + "\n".join(rows)
        
    # Build the prompt using standard string concatenation to avoid any f-string issues
    prompt = "Given the user's question and the SQL query result below, write a clear, concise answer in English. If the result is empty, say so.\n\n"
    prompt += f"User question: {question}\n\n"
    prompt += f"SQL query: {sql_query}\n\n"
    prompt += f"Query result:\n{result_str}\n\n"
    prompt += "Answer:\n"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
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
    """Connect to MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def execute_query(query: str) -> Tuple[List[str], List[List]]:
    """Execute query and return columns and results"""
    connection = connect_to_database()
    if not connection:
        return ["Error"], [["Database connection failed"]]
        
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        results = []
        for row in cursor.fetchall():
            results.append([str(cell) if cell is not None else "NULL" for cell in row])
        return columns, results
    except Exception as e:
        return ["Error"], [[str(e)]]
    finally:
        cursor.close()
        connection.close()

def create_interface():
    with gr.Blocks(title="RME PO Query Assistant rev.14 (Merged Table)") as interface:
        gr.Markdown(
            "# PO Follow-Up Query AI (rev.14, Merged Table)\n"
            "This AI assistant queries the merged PO table with all columns and terms.\n\n"
            "**Powered by Ollama Gemma3 running on your local GPU server.**"
        )
        
        with gr.Row():
            with gr.Column(scale=85):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about purchase orders, terms, items, suppliers, projects, etc...",
                    lines=2
                )
            with gr.Column(scale=15):
                submit_btn = gr.Button("Send", variant="primary")
                
        # New component to display detected entities
        with gr.Row():
            detected_entities = gr.Markdown(label="Detected Entities")
            
        with gr.Row():
            llm_answer = gr.Markdown(label="LLM Answer")
            
        with gr.Row():
            query_output = gr.Code(label="Generated SQL Query", language="sql")
            
        with gr.Row():
            results = gr.Dataframe(label="Query Results")
            
        with gr.Row():
            supplier_btn = gr.Button("Show Unique Suppliers", elem_id="supplier-btn")
            project_btn = gr.Button("Show Unique Projects", elem_id="project-btn")

        def on_submit(question):
            if not question.strip():
                return "No entities detected", "Please enter a question", "", gr.Dataframe(value=[["Please enter a question"]], headers=["Error"])
                
            try:
                answer, query, columns, results_data, entities = process_question(question)
                
                # Format detected entities for display
                entities_markdown = "### Detected Entities\n"
                
                # Display project and alternatives if available
                if entities["project"]:
                    entities_markdown += f"**Project:** {entities['project']} *(confidence: {entities['project_confidence']:.2f})*\n"
                    
                    # Show alternative project matches if any exist
                    if entities.get("project_alternatives") and len(entities["project_alternatives"]) > 0:
                        entities_markdown += "**Alternative projects:** "
                        entities_markdown += ", ".join([f"`{alt}`" for alt in entities["project_alternatives"]])
                        entities_markdown += "\n"
                else:
                    entities_markdown += "**Project:** None\n"
                    
                # Display supplier and alternatives if available
                if entities["supplier"]:
                    entities_markdown += f"**Supplier:** {entities['supplier']} *(confidence: {entities['supplier_confidence']:.2f})*\n"
                    
                    # Show alternative supplier matches if any exist
                    if entities.get("supplier_alternatives") and len(entities["supplier_alternatives"]) > 0:
                        entities_markdown += "**Alternative suppliers:** "
                        entities_markdown += ", ".join([f"`{alt}`" for alt in entities["supplier_alternatives"]])
                        entities_markdown += "\n"
                else:
                    entities_markdown += "**Supplier:** None\n"
                
                if query.startswith("Error") or columns[0] == "Error":
                    error_message = results_data[0][0] if columns[0] == "Error" else query
                    return entities_markdown, answer, query, gr.Dataframe(value=[[error_message]], headers=["Error"])
                    
                return entities_markdown, answer, query, gr.Dataframe(value=results_data, headers=columns)
                
            except Exception as e:
                return "Error detecting entities", f"Unexpected error: {str(e)}", "", gr.Dataframe(value=[[str(e)]], headers=["Error"])

        submit_btn.click(
            fn=on_submit,
            inputs=[question_input],
            outputs=[detected_entities, llm_answer, query_output, results]
        )
        
        question_input.submit(
            fn=on_submit,
            inputs=[question_input],
            outputs=[detected_entities, llm_answer, query_output, results]
        )
        
        supplier_btn.click(None, None, None, js="window.open('/suppliers', '_blank')")
        project_btn.click(None, None, None, js="window.open('/projects', '_blank')")
        
        gr.Examples(
            examples=[
                ["Show me all items for project Rabigh 2 (Mourjan)"],
                ["What are the terms for supplier الشركة المصرية"],
                ["List all POs for project MOC HQ at Diriyah-K0005"],
                ["Show all purchase orders with their terms"],
                ["Which item has the highest unit price in Ring Road project?"],
            ],
            inputs=question_input,
            outputs=[detected_entities, llm_answer, query_output, results],
            fn=on_submit
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
    names = get_unique_list("VENDOR_NAME")
    html = "<h2>Unique Supplier Names</h2><ul>" + "".join(f"<li>{n}</li>" for n in names) + "</ul>"
    return HTMLResponse(content=html)

@app.get("/projects", response_class=HTMLResponse)
def projects():
    names = get_unique_list("PROJECT_NAME")
    html = "<h2>Unique Project Names</h2><ul>" + "".join(f"<li>{n}</li>" for n in names) + "</ul>"
    return HTMLResponse(content=html)

if __name__ == "__main__":
    initialize_unique_lists()
    interface = create_interface()
    app = gr.mount_gradio_app(app, interface, path="/")
    webbrowser.open("http://localhost:7868")
    uvicorn.run(app, host="0.0.0.0", port=7868)
