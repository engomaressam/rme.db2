import gradio as gr
import mysql.connector
from mysql.connector import Error
import os
from openai import OpenAI
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
        
        # Initialize the OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        # Create a simple text prompt with table information
        system_message = """You are a SQL query generator. Generate MySQL queries based on natural language questions.
        The table name is 'po_followup_for_chatbot' and it has these columns:
        - po_number: the purchase order number (VARCHAR)
        - creation_date: when the purchase was made (DATE)
        - line_amount: how much we paid (DECIMAL)
        - vendor_name: who we bought from (VARCHAR)
        - project_name: which project this purchase was for (VARCHAR)
        
        Only use these columns in your query. Keep the query simple and focused on answering the question.
        Return ONLY the SQL query, with no explanations or markdown."""
        
        # Call OpenAI API with the latest API version
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
        # Clean the query by removing markdown code blocks if present
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
                    # Convert to integer and format with commas
                    formatted_row.append(f"{int(value):,}")
                else:
                    # Try to convert string numbers
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
    with gr.Blocks(title="RME PO Query Assistant") as interface:
        gr.Markdown("""
        # PO Follow-Up Query AI
        
        This AI assistant can help you query the Purchase Order database using natural language.
        **Note: This system uses a simplified table with essential PO information.**
        """)
        
        with gr.Row():
            with gr.Column(scale=85):
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about purchase orders...",
                    lines=2
                )
            with gr.Column(scale=15):
                submit_btn = gr.Button("Send", variant="primary")
        
        with gr.Row():
            query_output = gr.Code(label="Generated SQL Query", language="sql")
        
        with gr.Row():
            results = gr.Dataframe(label="Query Results")
        
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
                
                # The results are already formatted in the execute_query function
                # Just return them directly to the dataframe
                return query, gr.Dataframe(value=results, headers=columns)
            except Exception as e:
                return f"Unexpected error: {str(e)}", gr.Dataframe(value=[[str(e)]], headers=["Error"])
        
        # Set up event handler for the button
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input],
            outputs=[query_output, results]
        )
        
        # Also trigger on pressing Enter in the textbox
        question_input.submit(
            fn=on_submit,
            inputs=[question_input],
            outputs=[query_output, results]
        )
        
        # Add examples at the bottom that run automatically when clicked
        gr.Examples(
            examples=[
                ["Show me the top 10 vendors by total purchase amount"],
                ["What are the total purchases for each project in the last month?"],
                ["List the top 10 projects by total purchase amount"],
                ["What's the average purchase amount by vendor?"],
                ["Show me the purchase orders created yesterday"],
                ["What's the total amount of POs in May 2025?"]
            ],
            inputs=question_input,
            outputs=[query_output, results],
            fn=on_submit
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7866)
