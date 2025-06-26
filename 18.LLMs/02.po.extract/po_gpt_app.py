import os
import openai
import mysql.connector
from mysql.connector import Error
from datetime import datetime, timedelta
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()

def connect_to_db():
    """Connect to the MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='10.10.11.242',
            user='omar2',
            password='Omar_54321',
            database='RME_TEST'
        )
        return connection
    except Error as e:
        raise Exception(f"Database connection error: {e}")

def generate_sql(user_question):
    """Generate SQL from user question using OpenAI"""
    # Set OpenAI key
    openai.api_key = os.getenv('OPENAI_API_KEY')
    
    # Create prompt with table information
    prompt = f"""Based on this question: "{user_question}"

Generate a MySQL query to answer it.

The table name is 'po_followup_for_gpt' and it has these columns:
- po_number: the purchase order number (VARCHAR)
- creation_date: when the purchase was made (DATE)
- vendor_name: who we bought from (VARCHAR)
- line_amount: how much we paid (DECIMAL)
- project_name: which project this purchase was for (VARCHAR)

Only use these columns in your query. Keep the query simple and focused on answering the question."""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.1,
            max_tokens=100
        )
        
        query = response.choices[0].message.content.strip()
        # Clean the query by removing markdown code blocks
        query = query.replace('```sql', '').replace('```', '').strip()
        return query
    except Exception as e:
        return f"Error generating SQL: {str(e)}"

def execute_query(query):
    """Execute the SQL query and return results"""
    try:
        connection = connect_to_db()
        cursor = connection.cursor()
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Get column names
        column_names = [i[0] for i in cursor.description]
        
        # Format results as a list of dictionaries
        results = []
        for row in rows:
            result_dict = {}
            for i, col in enumerate(column_names):
                result_dict[col] = row[i]
            results.append(result_dict)
            
        cursor.close()
        connection.close()
        
        return results, column_names
    except Exception as e:
        return f"Error executing query: {str(e)}", []

def process_question(question):
    """Process user question, generate SQL, and return results"""
    if not question.strip():
        return "Please enter a question", "", []
    
    # Generate SQL query
    sql_query = generate_sql(question)
    
    # Execute query if SQL was generated successfully
    if sql_query and not sql_query.startswith("Error"):
        try:
            results, columns = execute_query(sql_query)
            
            # Format results for display
            if isinstance(results, str):  # Error message
                return sql_query, results, []
            else:
                # Create a formatted table
                table_data = []
                if results:
                    table_data = results
                return sql_query, "Query executed successfully", table_data
        except Exception as e:
            return sql_query, f"Error: {str(e)}", []
    else:
        return sql_query, "Failed to generate SQL query", []

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="PO Query Assistant") as demo:
        gr.Markdown("# PO Query Assistant")
        gr.Markdown("Ask questions about purchase orders in natural language")
        
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question", 
                    placeholder="e.g., Show me the top 5 vendors by purchase amount in the last month",
                    lines=3
                )
                submit_btn = gr.Button("Generate Answer")
            
        with gr.Row():
            with gr.Column():
                sql_output = gr.Textbox(label="Generated SQL Query")
                status_output = gr.Textbox(label="Status")
                
        with gr.Row():
            results_output = gr.Dataframe(label="Results")
            
        # Set up event handler
        submit_btn.click(
            fn=process_question,
            inputs=[question_input],
            outputs=[sql_output, status_output, results_output]
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
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(share=True)
