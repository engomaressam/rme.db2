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
    'schema': 'po_data'
}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:latest"

# PostgreSQL uses lowercase column names by default
NEW_TABLE = "po_followup_rev19"

# Column name mapping to handle case sensitivity in PostgreSQL
# In PostgreSQL, we need to use quotes for case-sensitive column names
COLUMN_MAP = {
    "project_name": "\"PROJECT_NAME\"",
    "vendor_name": "\"VENDOR_NAME\"",
    "po_num": "\"PO_NUM\"",
    "comments": "\"COMMENTS\"",
    "approved_date": "\"APPROVED_DATE\"",
    "uom": "\"UOM\"",
    "item_description": "\"ITEM_DESCRIPTION\"",
    "unit_price": "\"UNIT_PRICE\"",
    "quantity_received": "\"QUANTITY_RECEIVED\"",
    "line_amount": "\"LINE_AMOUNT\"",
    "terms": "\"TERMS\""
}

# Conversation history management
class ConversationManager:
    def __init__(self, max_history=5):
        self.conversation_history = []
        self.detected_entities = {}
        self.last_query_result = None
        self.last_sql_query = None
        self.max_history = max_history
        self.memory_was_cleared = False  # Flag to track if memory was explicitly cleared
    
    def add_interaction(self, question: str, answer: str, entities: Dict, sql_query: str = None, query_result: Dict = None):
        """Add a new interaction to the conversation history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # When adding a new interaction, we're no longer in a cleared state
        self.memory_was_cleared = False
        
        # Save important entities for context
        if entities.get("project"):
            self.detected_entities["project"] = entities["project"]
        if entities.get("supplier"):
            self.detected_entities["supplier"] = entities["supplier"]
            
        # Save query results for reference
        if query_result:
            self.last_query_result = query_result
        if sql_query:
            self.last_sql_query = sql_query
            
        # Add to history
        self.conversation_history.append({
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "entities": {k: v for k, v in entities.items() if k in ["project", "supplier"] and v is not None}
        })
        
        # Maintain max history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_context_for_llm(self) -> str:
        """Format conversation history for the LLM"""
        if not self.conversation_history:
            return ""
            
        context = "Previous conversation:\n"
        for i, interaction in enumerate(self.conversation_history):
            context += f"User: {interaction['question']}\n"
            context += f"Assistant: {interaction['answer']}\n"
        return context
    
    def get_active_entities(self) -> Dict:
        """Get detected entities from the conversation"""
        return self.detected_entities
    
    def reset(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.detected_entities = {}
        self.last_query_result = None
        self.last_sql_query = None
        self.memory_was_cleared = True  # Set flag to indicate memory was explicitly cleared

# Initialize conversation manager
CONVERSATION = ConversationManager()

# Fetch and cache unique projects and suppliers at startup
def fetch_unique_list(column: str) -> list:
    connection = None
    try:
        connection = connect_to_database()  # Using our PostgreSQL connection function
        if not connection:
            print(f"Error connecting to database when fetching unique {column}")
            return []
        
        # Get properly quoted column name for PostgreSQL
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
    UNIQUE_PROJECTS = fetch_unique_list("PROJECT_NAME")
    UNIQUE_SUPPLIERS = fetch_unique_list("VENDOR_NAME")

# Enhanced entity extraction and matching

def detect_entities_with_llm(question: str, use_history: bool = True) -> dict:
    """
    Use the LLM to detect project and supplier names mentioned in the question.
    Returns a dictionary with detected entities.
    If use_history is True, considers conversation history for context.
    """
    try:
        system_message = "You are an entity detection system. Your task is to identify project names and supplier names mentioned in questions about purchase orders."
        system_message += "\n\nAnalyze the question and extract ONLY project names or supplier names if they exist."
        system_message += "\n\nIf this is a follow-up question that refers to previously mentioned entities but doesn't explicitly name them, use the entities from the conversation history."
        system_message += "\n\nReturn your answer in JSON format with the following structure:"
        system_message += "\n{\"project\": \"project name or null if none mentioned\", \"supplier\": \"supplier name or null if none mentioned\"}"
        system_message += "\n\nIf a project or supplier is not mentioned in this question or previous context, use null (not empty string)."
        system_message += "\n\nExample 1: 'Show me all POs for Ring Road project'"
        system_message += "\nResponse: {\"project\": \"Ring Road\", \"supplier\": null}"
        system_message += "\n\nExample 2: 'What are the terms for supplier Siemens?'"
        system_message += "\nResponse: {\"project\": null, \"supplier\": \"Siemens\"}"
        system_message += "\n\nExample 3: [After talking about Ring Road project] 'What items were purchased for it?'"
        system_message += "\nResponse: {\"project\": \"Ring Road\", \"supplier\": null}"

        # Create message array with system instructions
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history context if available and requested and memory wasn't explicitly cleared
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
            
        # Once we've used the conversation history once after reset, clear the flag
        if CONVERSATION.memory_was_cleared:
            CONVERSATION.memory_was_cleared = False
        
        # Add the current question
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
        
        # Extract JSON from the response
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

def process_question(question: str, use_history: bool = True) -> Tuple[str, str, List[str], List[List], Dict]:
    """Process the question and return answer, query, columns, results, and detected entities"""
    detected_entities = {
        "project": None,
        "project_confidence": 0.0,
        "project_alternatives": [],
        "supplier": None,
        "supplier_confidence": 0.0,
        "supplier_alternatives": []
    }
    
    
    # Fallback to traditional entity extraction only if LLM response was invalid
    if llm_entities is None:
        project, project_confidence = extract_entity_from_question(question, "project", UNIQUE_PROJECTS)
        supplier, supplier_confidence = extract_entity_from_question(question, "supplier", UNIQUE_SUPPLIERS)
        
        detected_entities["project"] = project
        detected_entities["project_confidence"] = project_confidence
        detected_entities["supplier"] = supplier
        detected_entities["supplier_confidence"] = supplier_confidence
    
    # If we're using conversation history and no entities were detected in this question,
    # check if we have active entities from previous questions
    if use_history and (detected_entities["project"] is None or detected_entities["supplier"] is None):
        active_entities = CONVERSATION.get_active_entities()
        
        # Only use the conversation entity if none was detected in the current question
        # AND the conversation history isn't empty (to prevent using entities after clear)
        conversation_has_history = len(CONVERSATION.conversation_history) > 0
        
        if conversation_has_history and detected_entities["project"] is None and "project" in active_entities:
            detected_entities["project"] = active_entities["project"]
            detected_entities["project_confidence"] = 0.8  # Slightly lower confidence for inherited entities
            
        if conversation_has_history and detected_entities["supplier"] is None and "supplier" in active_entities:
            detected_entities["supplier"] = active_entities["supplier"]
            detected_entities["supplier_confidence"] = 0.8  # Slightly lower confidence for inherited entities
    
    # Generate SQL query based on the question and detected entities
    sql_query = generate_sql_query(question, detected_entities, use_history)
    
    # Add filters for detected entities to the query if not already included
    if detected_entities["project"] or detected_entities["supplier"]:
        has_where = "WHERE" in sql_query.upper()
        
        if detected_entities["project"] and not ("PROJECT_NAME" in sql_query and detected_entities["project"].lower() in sql_query.lower()):
            if has_where:
                sql_query = sql_query.replace("WHERE", f"WHERE \"PROJECT_NAME\" = '{detected_entities['project']}' AND ", 1)
                sql_query = sql_query.replace("where", f"where \"PROJECT_NAME\" = '{detected_entities['project']}' AND ", 1)
            else:
                sql_query += f" WHERE \"PROJECT_NAME\" = '{detected_entities['project']}'" 
                has_where = True
        
        if detected_entities["supplier"] and not ("VENDOR_NAME" in sql_query and detected_entities["supplier"].lower() in sql_query.lower()):
            if has_where:
                sql_query += f" AND \"VENDOR_NAME\" = '{detected_entities['supplier']}'" 
            else:
                sql_query += f" WHERE \"VENDOR_NAME\" = '{detected_entities['supplier']}'" 
    
    # Execute query
    try:
        # Execute query
        columns, results = execute_query(sql_query)
        query_result = {"columns": columns, "results": results}
        
        if columns[0] == "Error":
            # If there was an error, return it
            answer = results[0][0]
        else:
            # Generate natural language answer with conversation context
            answer = generate_natural_language_answer(question, sql_query, columns, results, use_history)
        
        # Update conversation history if using it
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
            
        # Explain all columns to the LLM
        system_message = "You are a SQL query generator. Generate PostgreSQL queries based on natural language questions.\n\n"
        system_message += "IMPORTANT: PostgreSQL is case-sensitive and column names are UPPERCASE in our database and must be quoted.\n"
        system_message += "Always use double quotes around column names like this: \"COLUMN_NAME\"\n\n"
        system_message += "The table name is 'po_followup_merged' and it has these columns:\n"
        system_message += "- id: auto-increment row id (INT)\n"
        system_message += "- \"PO_NUM\": purchase order number (VARCHAR)\n"
        system_message += "- \"COMMENTS\": comments about the PO (TEXT)\n"
        system_message += "- \"APPROVED_DATE\": date the PO was approved (DATE)\n"
        system_message += "- \"UOM\": unit of measure (VARCHAR)\n"
        system_message += "- \"ITEM_DESCRIPTION\": description of the item (TEXT)\n"
        system_message += "- \"UNIT_PRICE\": unit price for the item (DECIMAL)\n"
        system_message += "- \"QUANTITY_RECEIVED\": quantity received (DECIMAL)\n"
        system_message += "- \"LINE_AMOUNT\": total line amount (DECIMAL)\n"
        system_message += "- \"PROJECT_NAME\": project name (VARCHAR)\n"
        system_message += "- \"VENDOR_NAME\": supplier/vendor name (VARCHAR)\n"
        system_message += "- \"TERMS\": merged PO terms (TEXT)\n\n"
        system_message += "You can answer questions about purchase orders, terms, suppliers, projects, items, etc.\n\n"
        system_message += "Only use these columns in your query. Keep the query simple and focused on answering the question. "
        system_message += "Return ONLY the raw SQL query, with NO code formatting, NO markdown backticks, NO ```sql tags, and NO explanations."
        system_message += "\n\nImportant: If this is a follow-up question to a previous conversation, consider the context."
        
        # Create message array with system instructions
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history context if available and requested
        if use_history and CONVERSATION.conversation_history:
            context_message = "Previous conversation context:\n"
            
            # Add the last few interactions for context
            for interaction in CONVERSATION.conversation_history[-2:]:  # Use last 2 interactions
                context_message += f"User asked: {interaction['question']}\n"
                if CONVERSATION.last_sql_query:
                    context_message += f"SQL used: {CONVERSATION.last_sql_query}\n"
                
            # Add currently active entities from the conversation
            active_entities = CONVERSATION.get_active_entities()
            if active_entities:
                context_message += "\nActive entities from conversation:\n"
                for entity_type, entity_value in active_entities.items():
                    context_message += f"{entity_type}: {entity_value}\n"
                    
            messages.append({"role": "system", "content": context_message})
        
        # Add the detected entities for this question
        if detected_entities:
            entity_info = "\nEntities in current question:\n"
            if detected_entities.get("project"):
                entity_info += f"project_name: {detected_entities['project']}\n"
            if detected_entities.get("supplier"):
                entity_info += f"vendor_name: {detected_entities['supplier']}\n"
            if entity_info != "\nEntities in current question:\n":
                messages.append({"role": "system", "content": entity_info})
        
        # Add the current question
        messages.append({"role": "user", "content": f"Generate a PostgreSQL query to answer this question: {question}"})
        
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

def generate_natural_language_answer(question: str, sql_query: str, columns: list, results: list, use_history: bool = True) -> str:
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
    prompt += "Answer in clear, simple English. Be conversational. Don't repeat all values unless requested."
    
    if use_history and CONVERSATION.conversation_history:
        prompt += "\n\nThis is part of an ongoing conversation. Previous context:\n"
        # Add up to the last 2 conversation exchanges for context
        for interaction in CONVERSATION.conversation_history[-2:]:
            prompt += f"User: {interaction['question']}\n"
            prompt += f"Assistant: {interaction['answer']}\n"
        
        # Remind the model to refer to previously mentioned entities if relevant
        prompt += "\nIf this is a follow-up question, refer to entities mentioned in previous questions appropriately. "
        prompt += "For example, if a project was mentioned before but not in this question, still reference it by name."
    
    # Create message array with system instructions
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
    """Connect to PostgreSQL database"""
    try:
        connection = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        # Set search_path to our schema
        cursor = connection.cursor()
        cursor.execute(f"SET search_path TO {DB_CONFIG['schema']}")
        connection.commit()
        cursor.close()
        
        return connection
    except Error as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

def execute_query(query: str) -> Tuple[List[str], List[List]]:
    """Execute query and return columns and results"""
    connection = None
    try:
        connection = connect_to_database()
        if connection is None:
            return ["Error"], [["Could not connect to database"]]
            
        cursor = connection.cursor()
        cursor.execute(query)
        
        # For SELECT queries, fetch results and column names
        if query.strip().upper().startswith("SELECT"):
            columns = [desc[0] for desc in cursor.description]
            results = [list(row) for row in cursor.fetchall()]
        else:
            # For non-SELECT queries, just show status
            connection.commit()
            columns = ["Status"]
            results = [[f"Query executed successfully. Affected rows: {cursor.rowcount}"]] 
        
        # Close cursor and connection
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
    with gr.Blocks(title="RME PO Query Assistant rev.16 (PostgreSQL with Memory)") as interface:
        # Title only at the top - moved descriptive text to bottom
        gr.Markdown("# PO Follow-Up Query AI (rev.16, PostgreSQL with Memory)")
        
        # Two column layout - Chat on left, Entities on right
        with gr.Row():
            # Left column for chat interface
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="Conversation", height=400)
                
                with gr.Row():
                    with gr.Column(scale=8):
                        # Standard textbox for question input
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about purchase orders, terms, items, suppliers, projects, etc...",
                            lines=2
                        )
                    with gr.Column(scale=2):
                        submit_btn = gr.Button("Send", variant="primary")
                        clear_history_btn = gr.Button("Clear Memory", variant="secondary")
            
            # Right column for entity detection
            with gr.Column(scale=3):
                detected_entities = gr.Markdown("### Detected Entities\nNo entities detected yet.", label="Current Entities")
                
                with gr.Row():
                    supplier_btn = gr.Button("Show Suppliers", elem_id="supplier-btn")
                    project_btn = gr.Button("Show Projects", elem_id="project-btn")
        
        # SQL and results section (below the conversation)
        with gr.Accordion("SQL Query and Results", open=False):
            query_output = gr.Code(label="Current SQL Query", language="sql")
            results = gr.Dataframe(label="Query Results")
            
        # Informational text moved to bottom of the interface
        gr.Markdown(
            "### System Information\n"
            "This AI assistant queries the merged PO table with all columns and terms.\n\n"
            "- Using local PostgreSQL database with **861,403 PO records**\n"
            "- Ask questions about suppliers, terms, projects, etc.\n"
            "- Ask follow-up questions without repeating entities mentioned before\n"
            "- Navigation entity memory is applied to all follow-up questions\n"
        )
            
        def clear_history():
            # Completely reset the conversation memory
            global CONVERSATION
            # Create a new instance to ensure complete reset
            CONVERSATION = ConversationManager()
            print("Conversation memory has been completely cleared")
            print(f"Conversation state after reset: {CONVERSATION.conversation_history}")
            print(f"Entities after reset: {CONVERSATION.detected_entities}")
            
            # Return empty/reset values for all UI components
            return [], "### Detected Entities\nMemory has been reset. The chatbot will no longer remember previous questions and entities.", "", None
            
        def on_submit(question, chat_history):
            if not question.strip():
                return chat_history, question, "### Detected Entities\nNo query entered", "", None
                
            try:
                # Process the question with conversation history enabled
                answer, query, columns, results_data, entities = process_question(question, use_history=True)
                
                # Update the chat history with the new question and answer
                chat_history = chat_history + [(question, answer)]
                
                # Format detected entities for display in the right panel
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
                    return chat_history, "", entities_markdown, query, gr.Dataframe(value=[[error_message]], headers=["Error"])
                    
                # Create dataframe for results
                df = gr.Dataframe(value=results_data, headers=columns)
                return chat_history, "", entities_markdown, query, df
                
            except Exception as e:
                error_message = f"Unexpected error: {str(e)}"
                chat_history = chat_history + [(question, error_message)]
                return chat_history, "", "### Detected Entities\nError in entity detection", "", None

        # Set up event listeners
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, detected_entities, query_output, results]
        )
        
        # Add Enter key submission functionality
        question_input.submit(
            fn=on_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, detected_entities, query_output, results]
        )
        
        # Connect the clear history button
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
                "How many items were received for that project?",  # Example follow-up question
                "What were the terms for the last PO you showed me?",  # Another follow-up example
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
    # Use port 7869 for rev.16 to avoid conflict with rev.15 on port 7868
    webbrowser.open("http://localhost:7869")
    uvicorn.run(app, host="0.0.0.0", port=7869)
