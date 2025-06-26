import gradio as gr
import mysql.connector
from mysql.connector import Error
import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import webbrowser
import requests
import difflib

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:latest"

# Fetch and cache unique projects and suppliers at startup
def fetch_unique_list(column: str) -> list:
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        cursor.execute(f"SELECT DISTINCT {column} FROM po_followup_with_terms WHERE {column} IS NOT NULL AND {column} != '' ORDER BY {column}")
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
    UNIQUE_SUPPLIERS = fetch_unique_list("vendor_name")

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
        # Replace the closest substring in the question with the correct entity name
        words = entity.split()
        for w in words:
            if w.lower() in question.lower():
                rewritten_question = question.replace(w, entity)
                break
        else:
            rewritten_question = f"{question} ({entity})"
    # Continue as before
    query = generate_sql_query(rewritten_question)
    if query.startswith("Error"):
        return "", query, ["Error"], [["Failed to generate SQL query"]]
    columns, results = execute_query(query)
    answer = generate_natural_language_answer(rewritten_question, query, columns, results)
    return answer, query, columns, results

def generate_sql_query(question: str) -> str:
    try:
        system_message = """You are a SQL query generator. Generate MySQL queries based on natural language questions.\n\nThe table name is 'po_followup_with_terms' and it has these columns:\n- po_number: the purchase order number (VARCHAR)\n- creation_date: when the purchase was made (DATE)\n- line_amount: how much we paid (DECIMAL)\n- vendor_name: who we bought from (VARCHAR)\n- project_name: which project this purchase was for (VARCHAR)\n- terms: the payment and delivery agreements for the purchase order (TEXT)\n\nYou can answer questions about purchase order terms, such as:\n- What are the terms of PO number X?\n- What are the terms for supplier Y?\n- Show me the terms for all POs in project Z.\n\nOnly use these columns in your query. Keep the query simple and focused on answering the question. Return ONLY the SQL query, with no explanations or markdown."""
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
    prompt = f"""Given the user's question and the SQL query result below, write a clear, concise answer in English. If the result is empty, say so.\n\nUser question: {question}\n\nSQL query: {sql_query}\n\nQuery result:\n{result_str}\n\nAnswer:"""
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
                if i == 0 and columns[0].lower() in ['po_number', 'po_num']:
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
    with gr.Blocks(title="RME PO Query Assistant rev.12 (Fuzzy Project/Supplier Matching)") as interface:
        gr.Markdown("""
        # PO Follow-Up Query AI (rev.12, Fuzzy Project/Supplier Matching)
        This AI assistant can help you query the Purchase Order database using natural language, even if you don't type the exact project or supplier name!\n\n**Powered by Ollama Gemma3 running on your local GPU server.**
        """)
        with gr.Row():
            with gr.Column(scale=85):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about purchase orders, terms, etc...",
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
                return "Please enter a question", "", gr.Dataframe(value=[[]], headers=["Error"])
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
                ["Show me the POs for Abu Keir Bridge"],
                ["What are the terms for supplier الشركة المصرية"],
                ["List all POs for project PSP Master"],
                ["Show all purchase orders with their terms"],
            ],
            inputs=question_input,
            outputs=[llm_answer, query_output, results],
            fn=on_submit
        )
    return interface

app = FastAPI()

def get_unique_list(column: str):
    return fetch_unique_list(column)

@app.get("/suppliers", response_class=HTMLResponse)
def suppliers():
    names = get_unique_list("vendor_name")
    html = "<h2>Unique Supplier Names</h2><ul>" + "".join(f"<li>{n}</li>" for n in names) + "</ul>"
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
    webbrowser.open("http://localhost:7868")
    uvicorn.run(app, host="0.0.0.0", port=7868)
