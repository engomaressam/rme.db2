import gradio as gr
import mysql.connector
from mysql.connector import Error
import os
from openai import OpenAI
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn
import webbrowser

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

def connect_to_database():
    """Connect to MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def generate_sql_query(question: str) -> str:
    """Use GPT to convert natural language question to SQL query using simplified approach"""
    try:
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Error: OpenAI API key not found in environment variables"
        
        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        # Create a prompt that includes the new 'terms' column and instructs GPT to answer about terms
        system_message = """You are a SQL query generator. Generate MySQL queries based on natural language questions.\n\nThe table name is 'po_followup_with_terms' and it has these columns:\n- po_number: the purchase order number (VARCHAR)\n- creation_date: when the purchase was made (DATE)\n- line_amount: how much we paid (DECIMAL)\n- vendor_name: who we bought from (VARCHAR)\n- project_name: which project this purchase was for (VARCHAR)\n- terms: the payment and delivery agreements for the purchase order (TEXT)\n\nYou can answer questions about purchase order terms, such as:\n- What are the terms of PO number X?\n- What are the terms for supplier Y?\n- Show me the terms for all POs in project Z.\n\nOnly use these columns in your query. Keep the query simple and focused on answering the question. Return ONLY the SQL query, with no explanations or markdown."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Generate a MySQL query to answer this question: {question}"}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        # Extract and clean the query
        query = response.choices[0].message.content.strip()
        query = query.replace('```sql', '').replace('```', '').strip()
        return query
    except Exception as e:
        return f"Error generating query: {str(e)}"

def execute_query(query: str) -> Tuple[List[str], List[List]]:
    """Execute SQL query and return column names and results"""
    connection = connect_to_database()
    if not connection:
        return ["Error"], [["Could not connect to database"]]
    
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        raw_results = cursor.fetchall()
        
        # Format numeric values with commas and no decimals
        formatted_results = []
        for row in raw_results:
            formatted_row = []
            for i, value in enumerate(row):
                # Skip first column if it's a PO number
                if i == 0 and columns[0].lower() in ['po_number', 'po_num']:
                    formatted_row.append(value)
                    continue
                # Format numeric values
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

def process_question(question: str) -> Tuple[str, List[str], List[List]]:
    """Process a natural language question and return query and results"""
    query = generate_sql_query(question)
    if query.startswith("Error"):
        return query, ["Error"], [["Failed to generate SQL query"]]
    columns, results = execute_query(query)
    return query, columns, results

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="RME PO Query Assistant rev.08") as interface:
        gr.Markdown("""
        # PO Follow-Up Query AI (rev.08)
        This AI assistant can help you query the Purchase Order database using natural language.\n\n**Note:** This system uses a table with essential PO information, including payment/delivery terms.
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
            query_output = gr.Code(label="Generated SQL Query", language="sql")
        with gr.Row():
            results = gr.Dataframe(label="Query Results")
        with gr.Row():
            supplier_btn = gr.Button("Show Unique Suppliers", elem_id="supplier-btn")
            project_btn = gr.Button("Show Unique Projects", elem_id="project-btn")
        def on_submit(question):
            if not question.strip():
                return "Please enter a question", gr.Dataframe(value=[[]], headers=["Error"])
            try:
                query, columns, results = process_question(question)
                if query.startswith("Error") or columns[0] == "Error":
                    error_message = results[0][0] if columns[0] == "Error" else query
                    return query, gr.Dataframe(value=[[error_message]], headers=["Error"])
                return query, gr.Dataframe(value=results, headers=columns)
            except Exception as e:
                return f"Unexpected error: {str(e)}", gr.Dataframe(value=[[str(e)]], headers=["Error"])
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input],
            outputs=[query_output, results]
        )
        question_input.submit(
            fn=on_submit,
            inputs=[question_input],
            outputs=[query_output, results]
        )
        # Add JS to open new tabs for suppliers/projects
        supplier_btn.click(None, None, None, js="window.open('/suppliers', '_blank')")
        project_btn.click(None, None, None, js="window.open('/projects', '_blank')")
        gr.Examples(
            examples=[
                ["Show me the top 10 vendors by total purchase amount"],
                ["What are the terms of PO number 12345?"],
                ["List all POs for supplier X and show their terms"],
                ["What are the terms for project ABC?"],
                ["Show all purchase orders with their terms"],
                ["Show me the purchase orders created yesterday"],
                ["What's the total amount of POs in May 2025?"]
            ],
            inputs=question_input,
            outputs=[query_output, results],
            fn=on_submit
        )
    return interface

# FastAPI endpoints for unique suppliers and projects
app = FastAPI()

def get_unique_list(column: str):
    connection = connect_to_database()
    if not connection:
        return ["Could not connect to database"]
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT DISTINCT {column} FROM po_followup_with_terms ORDER BY {column}")
        results = cursor.fetchall()
        return [row[0] for row in results if row[0]]
    finally:
        cursor.close()
        connection.close()

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
    interface = create_interface()
    app = gr.mount_gradio_app(app, interface, path="/")
    # Open browser to Gradio main page
    webbrowser.open("http://localhost:7868")
    uvicorn.run(app, host="0.0.0.0", port=7868)
