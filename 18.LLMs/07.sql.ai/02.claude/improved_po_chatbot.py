import gradio as gr
import psycopg2
from psycopg2 import Error
import os
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Union
from dotenv import load_dotenv
import requests
import difflib
from datetime import datetime
import time
import traceback
import hashlib

# Load environment variables
load_dotenv()

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:latest"
OLLAMA_EXTRACTION_MODEL = "mistral:instruct"

# --- Enhanced Memory and Conversation Management ---
class ConversationMemory:
    def __init__(self):
        self.history = []
        self.active_entities = {"project": None, "supplier": None}
        self.last_query_results = None
        self.context_relevance_threshold = 3  # Number of turns to keep context active
        
    def add_turn(self, user_query, bot_response, detected_entities, sql_results=None):
        self.history.append({
            "user": user_query,
            "bot": bot_response,
            "entities": detected_entities.copy(),
            "timestamp": datetime.now(),
            "sql_results": sql_results
        })
        
        # Update active entities only if they were explicitly mentioned or detected
        for entity_type in ["project", "supplier"]:
            if detected_entities.get(entity_type) and detected_entities.get(f"{entity_type}_confidence", 0) > 0.6:
                self.active_entities[entity_type] = detected_entities[entity_type]
        
        # Keep only recent history to avoid memory bloat
        if len(self.history) > 10:
            self.history = self.history[-10:]
    
    def get_relevant_context(self, current_question: str) -> str:
        """Get relevant context from recent conversation history"""
        if not self.history:
            return ""
        
        # Get last few turns for context
        recent_history = self.history[-3:]
        context_parts = []
        
        for turn in recent_history:
            if turn['entities'].get('project') or turn['entities'].get('supplier'):
                context_parts.append(f"Previous: {turn['user']} (entities: {turn['entities']})")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def should_inherit_entity(self, entity_type: str, current_question: str) -> bool:
        """Determine if we should inherit an entity from context"""
        if not self.active_entities.get(entity_type):
            return False
        
        # Check if current question seems to be a follow-up
        follow_up_indicators = [
            "show me", "what about", "how many", "list", "give me",
            "what are", "tell me", "display", "get", "find"
        ]
        
        question_lower = current_question.lower()
        has_follow_up_pattern = any(indicator in question_lower for indicator in follow_up_indicators)
        
        # Don't inherit if question explicitly mentions different entities
        if entity_type == "project":
            if re.search(r'\b(?:project|for)\s+[\w\s]+', current_question):
                return False
        elif entity_type == "supplier":
            if re.search(r'\b(?:supplier|vendor|from)\s+[\w\s]+', current_question):
                return False
        
        return has_follow_up_pattern and len(self.history) <= self.context_relevance_threshold
    
    def clear(self):
        self.history.clear()
        self.active_entities = {"project": None, "supplier": None}
        self.last_query_results = None

CONVERSATION = ConversationMemory()

# --- Unique List Fetching ---
def fetch_unique_list(column):
    try:
        connection = psycopg2.connect(**{k: v for k, v in DB_CONFIG.items() if k != 'schema'})
        cursor = connection.cursor()
        cursor.execute(f"SELECT DISTINCT {column} FROM po_followup_rev17 WHERE {column} IS NOT NULL AND {column} != ''")
        rows = cursor.fetchall()
        return sorted([row[0].strip() for row in rows if row[0] and str(row[0]).strip()])
    except Exception as e:
        print(f"Error fetching unique list for {column}: {e}")
        return []
    finally:
        if connection:
            connection.close()

# Cache unique values
UNIQUE_PROJECTS = fetch_unique_list("project_name")
UNIQUE_SUPPLIERS = fetch_unique_list("vendor")

print(f"Loaded {len(UNIQUE_PROJECTS)} unique projects and {len(UNIQUE_SUPPLIERS)} unique suppliers")

# --- Enhanced Entity Extraction ---
def detect_entities_with_llm(question: str, context: str = "") -> dict:
    """Enhanced LLM-based entity detection with better prompting"""
    try:
        # Provide sample entities for better matching
        sample_projects = UNIQUE_PROJECTS[:10] if UNIQUE_PROJECTS else []
        sample_suppliers = UNIQUE_SUPPLIERS[:10] if UNIQUE_SUPPLIERS else []
        
        system_prompt = f"""You are an expert entity extraction system for purchase order queries.

Your task: Extract project names and supplier/vendor names from user questions.

Available entities (examples):
Projects: {sample_projects}
Suppliers: {sample_suppliers}

Rules:
1. Look for explicit mentions of projects or suppliers in the question
2. Consider context from previous conversation
3. Return exact matches from available entities when possible
4. Use fuzzy matching for partial matches
5. Return null if no clear entity is mentioned

Context: {context}

Return ONLY a JSON object with this exact format:
{{"project": "exact_project_name_or_null", "supplier": "exact_supplier_name_or_null", "confidence": 0.8}}

Examples:
Q: "Show me POs for Ring Road project"
A: {{"project": "Ring Road", "supplier": null, "confidence": 0.9}}

Q: "What orders do we have from Siemens?"
A: {{"project": null, "supplier": "Siemens", "confidence": 0.9}}

Q: "How many items were delivered?" (after discussing Ring Road)
A: {{"project": "Ring Road", "supplier": null, "confidence": 0.7}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": 0.1  # Lower temperature for more consistent extraction
        }
        
        response = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        llm_response = result['choices'][0]['message']['content'].strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            try:
                entities = json.loads(json_match.group(0))
                return {
                    "project": entities.get("project") if entities.get("project") != "null" else None,
                    "supplier": entities.get("supplier") if entities.get("supplier") != "null" else None,
                    "confidence": entities.get("confidence", 0.5)
                }
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}, Raw response: {llm_response}")
        
        print(f"No valid JSON found in LLM response: {llm_response}")
        return {"project": None, "supplier": None, "confidence": 0.0}
        
    except Exception as e:
        print(f"Error in LLM entity detection: {str(e)}")
        return {"project": None, "supplier": None, "confidence": 0.0}

def fuzzy_match_entity(candidate: str, entity_list: list, threshold: float = 0.6) -> Tuple[Optional[str], float]:
    """Improved fuzzy matching with multiple strategies"""
    if not candidate or not entity_list:
        return None, 0.0
    
    candidate_clean = candidate.strip().lower()
    best_match = None
    best_score = 0.0
    
    for entity in entity_list:
        entity_clean = entity.strip().lower()
        
        # Strategy 1: Exact match
        if candidate_clean == entity_clean:
            return entity, 1.0
        
        # Strategy 2: Substring match
        if candidate_clean in entity_clean or entity_clean in candidate_clean:
            score = min(len(candidate_clean), len(entity_clean)) / max(len(candidate_clean), len(entity_clean))
            if score > best_score:
                best_match, best_score = entity, score
        
        # Strategy 3: Fuzzy ratio
        ratio = difflib.SequenceMatcher(None, candidate_clean, entity_clean).ratio()
        if ratio > best_score:
            best_match, best_score = entity, ratio
        
        # Strategy 4: Token-based matching for multi-word entities
        candidate_tokens = set(candidate_clean.split())
        entity_tokens = set(entity_clean.split())
        if candidate_tokens and entity_tokens:
            intersection = candidate_tokens.intersection(entity_tokens)
            union = candidate_tokens.union(entity_tokens)
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard > best_score:
                best_match, best_score = entity, jaccard
    
    return (best_match, best_score) if best_score >= threshold else (None, 0.0)

def extract_entities_from_question(question: str) -> Dict:
    """Main entity extraction function combining LLM and fuzzy matching"""
    
    # Get context from conversation history
    context = CONVERSATION.get_relevant_context(question)
    
    # First, try LLM extraction
    llm_result = detect_entities_with_llm(question, context)
    
    detected_entities = {
        "project": None,
        "project_confidence": 0.0,
        "supplier": None, 
        "supplier_confidence": 0.0,
        "method": "none"
    }
    
    # Process LLM results with fuzzy matching
    if llm_result.get("project"):
        matched_project, conf = fuzzy_match_entity(llm_result["project"], UNIQUE_PROJECTS)
        if matched_project:
            detected_entities["project"] = matched_project
            detected_entities["project_confidence"] = conf * llm_result.get("confidence", 0.5)
            detected_entities["method"] = "llm+fuzzy"
    
    if llm_result.get("supplier"):
        matched_supplier, conf = fuzzy_match_entity(llm_result["supplier"], UNIQUE_SUPPLIERS)
        if matched_supplier:
            detected_entities["supplier"] = matched_supplier
            detected_entities["supplier_confidence"] = conf * llm_result.get("confidence", 0.5)
            detected_entities["method"] = "llm+fuzzy"
    
    # Fallback: Direct fuzzy matching on the question
    if not detected_entities["project"]:
        project_match, conf = fuzzy_match_entity(question, UNIQUE_PROJECTS, threshold=0.4)
        if project_match:
            detected_entities["project"] = project_match
            detected_entities["project_confidence"] = conf
            detected_entities["method"] = "direct_fuzzy"
    
    if not detected_entities["supplier"]:
        supplier_match, conf = fuzzy_match_entity(question, UNIQUE_SUPPLIERS, threshold=0.4)
        if supplier_match:
            detected_entities["supplier"] = supplier_match
            detected_entities["supplier_confidence"] = conf
            detected_entities["method"] = "direct_fuzzy"
    
    # Apply inheritance logic for follow-up questions
    for entity_type in ["project", "supplier"]:
        if not detected_entities[entity_type] and CONVERSATION.should_inherit_entity(entity_type, question):
            inherited_entity = CONVERSATION.active_entities[entity_type]
            detected_entities[entity_type] = inherited_entity
            detected_entities[f"{entity_type}_confidence"] = 0.8  # Medium confidence for inherited
            detected_entities["method"] = "inherited"
            print(f"[INHERITED] Using {entity_type}: {inherited_entity}")
    
    return detected_entities

# --- SQL Query Generation ---
def build_sql_query(detected_entities: Dict, question: str) -> str:
    """Build SQL query based on detected entities and question intent"""
    
    base_query = "SELECT * FROM po_followup_rev17"
    conditions = []
    
    # Add entity filters
    if detected_entities.get("project"):
        conditions.append(f"project_name = '{detected_entities['project']}'")
    
    if detected_entities.get("supplier"):
        conditions.append(f"vendor = '{detected_entities['supplier']}'")
    
    # Analyze question for additional filters or aggregations
    question_lower = question.lower()
    
    # Status filters
    if "open" in question_lower and "po" in question_lower:
        conditions.append("po_status = 'Open'")
    elif "closed" in question_lower and "po" in question_lower:
        conditions.append("po_status = 'Closed'")
    
    # Build WHERE clause
    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)
    
    # Determine if this is an aggregation query
    if any(word in question_lower for word in ["count", "how many", "total", "sum"]):
        if "amount" in question_lower or "cost" in question_lower or "value" in question_lower:
            return f"SELECT COUNT(*) as po_count, SUM(amount) as total_amount, AVG(amount) as avg_amount FROM po_followup_rev17{where_clause}"
        else:
            return f"SELECT COUNT(*) as total_count FROM po_followup_rev17{where_clause}"
    
    # Default: return detailed records with limit
    return f"{base_query}{where_clause} ORDER BY approved_date DESC LIMIT 10"

# --- Answer Generation ---
def generate_answer_with_llm(question: str, sql_results: List, detected_entities: Dict) -> str:
    """Generate natural language answer from SQL results"""
    if not sql_results:
        entity_context = []
        if detected_entities.get("project"):
            entity_context.append(f"project '{detected_entities['project']}'")
        if detected_entities.get("supplier"):
            entity_context.append(f"supplier '{detected_entities['supplier']}'")
        
        context_str = " and ".join(entity_context) if entity_context else "the specified criteria"
        return f"No purchase orders found for {context_str}. Please check if the project or supplier names are correct."
    
    # Prepare data summary for LLM
    if len(sql_results) > 5:
        sample_results = sql_results[:5]
        data_summary = f"Found {len(sql_results)} records. Here are the first 5:\n"
    else:
        sample_results = sql_results
        data_summary = f"Found {len(sql_results)} records:\n"
    
    # Format results for LLM
    if sql_results and isinstance(sql_results[0], tuple) and len(sql_results[0]) > 5:
        # Detailed records
        for i, row in enumerate(sample_results):
            data_summary += f"{i+1}. PO: {row[2]}, Project: {row[0]}, Supplier: {row[1]}, Status: {row[3]}, Amount: {row[10]} {row[11]}\n"
    else:
        # Aggregated results
        data_summary += str(sample_results)
    
    try:
        system_prompt = """You are a helpful assistant that analyzes purchase order data and provides clear, concise answers.

Rules:
1. Answer the user's question directly based on the provided data
2. Use natural language and be conversational
3. Include relevant numbers, amounts, and key details
4. If showing multiple items, summarize the key points
5. Keep the response focused and not too long
6. Format monetary amounts clearly with currency symbols

Data columns: project_name, vendor, po_num, po_status, approved_date, po_comments, description, uom, unit_price, currency, amount, term, qty_delivered"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\n\nData: {data_summary}\n\nPlease provide a clear answer to the question based on this data."}
        ]
        
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "temperature": 0.3
        }
        
        response = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
        
    except Exception as e:
        print(f"Error generating answer with LLM: {e}")
        # Fallback to simple summary
        return f"Found {len(sql_results)} purchase orders matching your query. Here's a summary of the key information from the results."

# --- Main Processing Function ---
def process_question(question: str) -> Tuple[str, str, Dict]:
    """Main function to process user questions"""
    
    if not question.strip():
        return "Please ask a question about purchase orders.", "", {}
    
    try:
        # Step 1: Extract entities
        print(f"[PROCESSING] Question: {question}")
        detected_entities = extract_entities_from_question(question)
        print(f"[ENTITIES] Detected: {detected_entities}")
        
        # Step 2: Build and execute SQL query
        sql_query = build_sql_query(detected_entities, question)
        print(f"[SQL] Query: {sql_query}")
        
        # Execute query
        results = []
        try:
            connection = psycopg2.connect(**{k: v for k, v in DB_CONFIG.items() if k != 'schema'})
            cursor = connection.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            print(f"[RESULTS] Found {len(results)} records")
        except Exception as e:
            print(f"SQL execution error: {e}")
            return f"Database error: {str(e)}", sql_query, detected_entities
        finally:
            if connection:
                connection.close()
        
        # Step 3: Generate natural language answer
        answer = generate_answer_with_llm(question, results, detected_entities)
        
        # Step 4: Update conversation memory
        CONVERSATION.add_turn(question, answer, detected_entities, results)
        
        return answer, sql_query, detected_entities
        
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        print(f"[ERROR] {error_msg}")
        traceback.print_exc()
        return error_msg, "", {}

# --- Gradio Interface ---
def create_gradio_interface():
    """Create Gradio interface for the chatbot"""
    
    def chat_interface(message, history):
        if message.lower().strip() == "/clear":
            CONVERSATION.clear()
            return "Memory cleared. You can start a new conversation.", ""
        
        answer, sql_query, entities = process_question(message)
        
        # Format response with debug info
        debug_info = f"**Detected Entities:** {entities}\n**SQL Query:** `{sql_query}`"
        
        return answer, debug_info
    
    # Create interface
    with gr.Blocks(title="PO Chatbot") as demo:
        gr.Markdown("# Purchase Order Chatbot")
        gr.Markdown("Ask questions about purchase orders, projects, and suppliers. Type '/clear' to reset conversation memory.")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=400)
                msg = gr.Textbox(
                    placeholder="Ask about purchase orders... (e.g., 'Show me all POs for Ring Road project')",
                    label="Your Question"
                )
                
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear Chat")
            
            with gr.Column(scale=1):
                debug_output = gr.Markdown(label="Debug Info")
        
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, ""
            
            answer, debug_info = chat_interface(message, chat_history)
            chat_history.append((message, answer))
            return chat_history, debug_info
        
        submit.click(respond, [msg, chatbot], [chatbot, debug_output])
        msg.submit(respond, [msg, chatbot], [chatbot, debug_output])
        clear.click(lambda: ([], ""), outputs=[chatbot, debug_output])
    
    return demo

# --- Main Execution ---
if __name__ == "__main__":
    # Test the system
    test_questions = [
        "Show me all POs for Ring Road project",
        "What suppliers do we work with?",
        "How many open purchase orders do we have?",
        "What's the total amount for Siemens orders?"
    ]
    
    print("Testing the system...")
    for question in test_questions:
        print(f"\n--- Testing: {question} ---")
        answer, sql, entities = process_question(question)
        print(f"Answer: {answer}")
        print(f"Entities: {entities}")
        print("-" * 50)
    
    # Launch Gradio interface
    print("\nLaunching Gradio interface...")
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True)