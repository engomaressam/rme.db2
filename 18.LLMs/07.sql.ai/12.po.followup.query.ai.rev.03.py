import gradio as gr
import mysql.connector
from mysql.connector import Error
import os
import openai
from typing import Dict, List, Tuple
from dotenv import load_dotenv

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
        
        openai.api_key = api_key
        
        # Create a simple text prompt with table information
        prompt = f"""Based on this question: "{question}"

Generate a MySQL query to answer it.

The table name is 'po_followup_for_gpt' and it has these columns:
- po_number: the purchase order number (VARCHAR)
- creation_date: when the purchase was made (DATE)
- line_amount: how much we paid (DECIMAL)
- vendor_name: who we bought from (VARCHAR)
- project_name: which project this purchase was for (VARCHAR)

Only use these columns in your query. Keep the query simple and focused on answering the question."""
        
        # Call OpenAI API with the old-style API (v0.28.0)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            max_tokens=100
        )
        
        # Extract and clean the query
        query = response.choices[0].message.content.strip()
        # Clean the query by removing markdown code blocks
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
        results = cursor.fetchall()
        return columns, results
    except Error as e:
        return ["Error"], [[str(e)]]
    finally:
        cursor.close()
        connection.close()

def process_question(question: str) -> Tuple[str, List[str], List[List]]:
    """Process a natural language question and return query and results"""
    # Generate SQL query
    query = generate_sql_query(question)
    
    # Check if query generation was successful
    if query.startswith("Error"):
        return query, ["Error"], [["Failed to generate SQL query"]]
    
    # Execute query and get results
    columns, results = execute_query(query)
    
    return query, columns, results

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="PO Follow-Up Query AI") as interface:
        gr.Markdown("""
        # PO Follow-Up Query AI (Rev 03)
        
        This AI assistant can help you query the Purchase Order database using natural language.
        **Note: This system uses a simplified table with essential PO information.**
        
        Example questions:
        - Show me all POs from last month
        - What are the top 10 highest value POs?
        - Which vendors have the most purchases?
        - Show me projects with purchases over $100,000
        """)
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about purchase orders...",
                lines=2
            )
        
        with gr.Row():
            submit_btn = gr.Button("Get Answer", variant="primary")
        
        with gr.Row():
            query_output = gr.Code(label="Generated SQL Query", language="sql")
        
        with gr.Row():
            results_output = gr.Dataframe(
                label="Query Results",
                interactive=False
            )
        
        def on_submit(question):
            if not question.strip():
                return "Please enter a question", gr.Dataframe(value=[[]], headers=["Error"])
            
            try:
                # Generate and execute query
                query, columns, results = process_question(question)
                
                # Check for errors
                if query.startswith("Error") or columns[0] == "Error":
                    error_message = results[0][0] if columns[0] == "Error" else query
                    return query, gr.Dataframe(value=[[error_message]], headers=["Error"])
                
                return query, gr.Dataframe(value=results, headers=columns)
            except Exception as e:
                return f"Unexpected error: {str(e)}", gr.Dataframe(value=[[str(e)]], headers=["Error"])
        
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input],
            outputs=[query_output, results_output]
        )
        
        # Add example questions
        gr.Examples(
            examples=[
                ["Show me the top 10 vendors by total purchase amount"],
                ["What are the total purchases for each project in the last month?"],
                ["List all purchases made for the PLAYA Resorts Project"],
                ["What's the average purchase amount by vendor?"],
                ["Show me the purchase orders created yesterday"]
            ],
            inputs=question_input
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="localhost", server_port=7861)
