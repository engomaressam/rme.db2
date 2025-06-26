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

# LLM-aided entity extraction and fuzzy matching
def extract_entity_from_question(question: str, entity_type: str, entity_list: list) -> Optional[str]:
    system_message = f"""You are an assistant that helps extract the most likely {entity_type} name from a user's question.\n\nHere is a list of all valid {entity_type} names:\n{chr(10).join(entity_list)}\n\nIf the question refers to a {entity_type}, output the closest matching name from the list above (exactly as written). If not, output NONE."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        "stream": False
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        response_data = resp.json()
        guess = response_data['choices'][0]['message']['content'].strip()
        if guess.upper() == "NONE":
            return None
        # Fuzzy match to the closest in the list
        close = difflib.get_close_matches(guess, entity_list, n=1, cutoff=0.6)
        return close[0] if close else None
    except Exception as e:
        print(f"Error extracting {entity_type}: {e}")
        return None

def process_question(question: str) -> Tuple[str, str, list, list]:
    # Try to extract project or supplier from the question
    entity = None
    entity_type = None
    project_guess = extract_entity_from_question(question, "project", UNIQUE_PROJECTS)
    if project_guess:
        entity = project_guess
        entity_type = "project"
    else:
        supplier_guess = extract_entity_from_question(question, "supplier", UNIQUE_SUPPLIERS)
        if supplier_guess:
            entity = supplier_guess
            entity_type = "supplier"
    # If an entity was found, rewrite the question with the exact name
    rewritten_question = question
    if entity:
        words = entity.split()
        for w in words:
            if w.lower() in question.lower():
                rewritten_question = question.replace(w, entity)
                break
        else:
            rewritten_question = f"{question} ({entity})"
    query = generate_sql_query(rewritten_question)
    if query.startswith("Error"):
        return "", query, ["Error"], [["Failed to generate SQL query"]]
    columns, results = execute_query(query)
    answer = generate_natural_language_answer(rewritten_question, query, columns, results)
    return answer, query, columns, results

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
        system_message += "Return ONLY the SQL query, with no explanations or markdown."
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
        response_data = resp.json()
        query = response_data['choices'][0]['message']['content'].strip()
        query = query.replace('```sql', '').replace('```', '').strip()
        return query
    except Exception as e:
        return f"Error generating query: {str(e)}"

def generate_natural_language_answer(question: str, sql_query: str, columns: list, results: list) -> str:
    if not results or columns == ["Error"]:
        result_str = "No results."
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
            {"role": "system", "content": "You are a helpful assistant who explains database query results in clear English."},
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        response_data = resp.json()
        answer = response_data['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def connect_to_database():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def execute_query(query: str) -> Tuple[list, list]:
    connection = connect_to_database()
    if not connection:
        return ["Error"], [["Could not connect to database"]]
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        raw_results = cursor.fetchall()
        formatted_results = []
        for row in raw_results:
            formatted_row = []
            for i, value in enumerate(row):
                if i == 0 and columns[0].lower() == 'id':
                    formatted_row.append(value)
                    continue
                if isinstance(value, (int, float)):
                    formatted_row.append(f"{int(value):,}")
                else:
                    if isinstance(value, str) and '.' in value:
                        try:
                            num_value = float(value.replace(',', ''))
                            formatted_row.append(f"{int(num_value):,}")
                        except:
                            formatted_row.append(value)
                    else:
                        formatted_row.append(value)
            formatted_results.append(formatted_row)
        return columns, formatted_results
    except Error as e:
        return ["Error"], [[str(e)]]
    finally:
        cursor.close()
        connection.close()

def create_interface():
    with gr.Blocks(title="RME PO Query Assistant rev.13 (Merged Table)") as interface:
        gr.Markdown(
            "# PO Follow-Up Query AI (rev.13, Merged Table)\n"
            "This AI assistant queries the new merged PO table with all columns and merged terms.\n\n"
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
                return "Please enter a question", "", gr.Dataframe(value=[["Please enter a question"]], headers=["Error"])
            try:
                answer, query, columns, results = process_question(question)
                if query.startswith("Error") or columns[0] == "Error":
                    error_message = results[0][0] if columns[0] == "Error" else query
                    return answer, query, gr.Dataframe(value=[[error_message]], headers=["Error"])
                return answer, query, gr.Dataframe(value=results, headers=columns)
            except Exception as e:
                return f"Unexpected error: {str(e)}", "", gr.Dataframe(value=[[str(e)]], headers=["Error"])

        submit_btn.click(
            fn=on_submit,
            inputs=[question_input],
            outputs=[llm_answer, query_output, results]
        )
        question_input.submit(
            fn=on_submit,
            inputs=[question_input],
            outputs=[llm_answer, query_output, results]
        )
        supplier_btn.click(None, None, None, js="window.open('/suppliers', '_blank')")
        project_btn.click(None, None, None, js="window.open('/projects', '_blank')")
        gr.Examples(
            examples=[
                ["Show me all items for project Rabigh 2 (Mourjan)"],
                ["What are the terms for supplier \u0627\u0644\u0634\u0631\u0643\u0629 \u0627\u0644\u0645\u0635\u0631\u064a\u0629"],
                ["List all POs for project MOC HQ at Diriyah-K0005"],
                ["Show all purchase orders with their terms"],
            ],
            inputs=question_input,
            outputs=[llm_answer, query_output, results],
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
