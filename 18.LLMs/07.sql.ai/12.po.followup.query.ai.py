import gradio as gr
import mysql.connector
from mysql.connector import Error
import os
import openai
from typing import Dict, List, Tuple

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

def get_table_schema() -> str:
    """Get the schema of RME_PO_Follow_Up_Report table"""
    connection = connect_to_database()
    if not connection:
        return "Error: Could not connect to database"
    
    cursor = connection.cursor()
    try:
        # Get column information
        cursor.execute("DESCRIBE RME_PO_Follow_Up_Report")
        columns = cursor.fetchall()
        
        # Format schema information
        schema = "Table: RME_PO_Follow_Up_Report\nColumns:\n"
        for col in columns:
            schema += f"- {col[0]} ({col[1]})\n"
        
        return schema
    except Error as e:
        return f"Error getting schema: {e}"
    finally:
        cursor.close()
        connection.close()

# OpenAI API key - Replace this with your actual key
OPENAI_API_KEY = "removed

def generate_sql_query(question: str, schema: str) -> Tuple[str, str]:
    """Use GPT to convert natural language question to SQL query"""
    try:
        if not OPENAI_API_KEY:
            return "Error: OpenAI API key not set", ""
        openai.api_key = OPENAI_API_KEY
        
        prompt = f"""Given this database schema:
{schema}

For this question: "{question}"

Provide two things:
1. A MySQL query to answer the question (must use proper syntax and only columns from schema)
2. A template to describe the results (use placeholders like {total_rows} for actual numbers)

Format your response exactly like this:
---QUERY---
[Your SQL query here]
---TEMPLATE---
[Your result description template here]

Rules:
1. Query must use only columns from schema
2. Limit results to 100 rows
3. Add helpful comments in query
4. Template should be 1-2 sentences with placeholders"""

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Your task is to convert natural language questions into valid MySQL queries. Only return the SQL query itself, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract query and template
        query_part = content.split('---TEMPLATE---')[0].split('---QUERY---')[1].strip()
        template_part = content.split('---TEMPLATE---')[1].strip()
        
        return query_part, template_part
    except Exception as e:
        return f"Error generating query: {e}"

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

def process_question(question: str) -> Tuple[str, str, List[str], List[List], str]:
    """Process a natural language question and return query, results and description"""
    # Get schema
    schema = get_table_schema()
    
    # Generate SQL query and template
    query, template = generate_sql_query(question, schema)
    
    # Execute query and get results
    columns, results = execute_query(query)
    
    # Format description with actual numbers
    description = template
    try:
        description = template.format(
            total_rows=len(results),
            # Add any other placeholders you want to support
        )
    except:
        # If template formatting fails, use template as is
        pass
    
    return query, description, columns, results

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(title="PO Follow-Up Query AI") as interface:
        gr.Markdown("""
        # PO Follow-Up Query AI
        
        This AI assistant can help you query the PO Follow-Up Report database using natural language.
        **Note: This system only has access to the RME_PO_Follow_Up_Report table.**
        
        Example questions:
        - Show me all POs from last month
        - What are the top 10 highest value POs?
        - How many POs are currently pending?
        """)
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about the PO Follow-Up data...",
                lines=2
            )
        
        with gr.Row():
            submit_btn = gr.Button("Get Answer", variant="primary")
        
        with gr.Row():
            description_output = gr.Markdown(label="Result Description")
        
        with gr.Row():
            query_output = gr.Code(label="Generated SQL Query", language="sql")
        
        with gr.Row():
            results_output = gr.Dataframe(
                label="Query Results",
                interactive=False
            )
        
        def on_submit(question):
            try:
                # First get the table schema
                schema = get_table_schema()
                if schema.startswith('Error'):
                    return [
                        f"Database Error: {schema}",
                        "",
                        gr.Dataframe(value=[[]], headers=[])
                    ]

                # Generate query
                try:
                    query, template = generate_sql_query(question, schema)
                except Exception as e:
                    return [
                        f"Error generating query: {str(e)}",
                        "",
                        gr.Dataframe(value=[[]], headers=[])
                    ]

                # Execute query
                try:
                    columns, results = execute_query(query)
                    if columns[0] == 'Error':
                        return [
                            f"Error executing query: {results[0][0]}",
                            query,
                            gr.Dataframe(value=[[]], headers=[])
                        ]
                except Exception as e:
                    return [
                        f"Error executing query: {str(e)}",
                        query,
                        gr.Dataframe(value=[[]], headers=[])
                    ]

                # Format description
                try:
                    description = template.format(
                        total_rows=len(results)
                    )
                except:
                    description = template

                return [
                    description,
                    query,
                    gr.Dataframe(value=results, headers=columns)
                ]
            except Exception as e:
                return [
                    f"Unexpected error: {str(e)}",
                    "",
                    gr.Dataframe(value=[[]], headers=[])
                ]
        
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input],
            outputs=[description_output, query_output, results_output]
        )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7861)
