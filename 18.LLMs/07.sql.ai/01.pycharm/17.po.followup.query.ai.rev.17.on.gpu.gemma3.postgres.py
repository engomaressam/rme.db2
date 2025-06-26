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
NEW_TABLE = "po_followup_merged"

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

# System metadata tracking
SYSTEM_STATS = {
    "queries_processed": 0,
    "complex_queries": 0,
    "failed_queries": 0,
    "query_time_avg": 0,
    "total_query_time": 0,
    "version": "17.0"
}

# Query complexity classification thresholds
QUERY_COMPLEXITY = {
    "SIMPLE": 0,
    "MODERATE": 1,
    "COMPLEX": 2,
    "VERY_COMPLEX": 3
}

# Enhanced Conversation History Management
class ConversationManager:
    def __init__(self, max_history=5):
        self.conversation_history = []
        self.detected_entities = {}
        self.entity_relationships = {}  # New: Track relationships between entities
        self.last_query_result = None
        self.last_sql_query = None
        self.last_question = None
        self.max_history = max_history
        self.memory_was_cleared = False  # Flag to track if memory was explicitly cleared
        self.query_complexity_history = []  # Track complexity of previous queries
        self.clarification_needed = False  # Flag for when clarification is needed
        self.suggested_clarification = None  # Store suggested clarification question

    def add_interaction(self, question: str, answer: str, entities: Dict, 
                     sql_query: str = None, query_result: Dict = None, 
                     complexity: int = 0):
        """Add a new interaction to the conversation history with enhanced metadata"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # When adding a new interaction, we're no longer in a cleared state
        self.memory_was_cleared = False
        self.last_question = question

        # Save important entities for context
        for entity_type, entity_value in entities.items():
            if entity_type in ["project", "supplier"] and entity_value:
                self.detected_entities[entity_type] = entity_value

                # Track relationships between entities if we have multiple entity types
                if "project" in self.detected_entities and "supplier" in self.detected_entities:
                    project = self.detected_entities["project"]
                    supplier = self.detected_entities["supplier"]

                    # Initialize relationship tracking dictionaries if they don't exist
                    if "projects_by_supplier" not in self.entity_relationships:
                        self.entity_relationships["projects_by_supplier"] = {}
                    if "suppliers_by_project" not in self.entity_relationships:
                        self.entity_relationships["suppliers_by_project"] = {}

                    # Record relationship: this supplier works on this project
                    if supplier not in self.entity_relationships["projects_by_supplier"]:
                        self.entity_relationships["projects_by_supplier"][supplier] = []
                    if project not in self.entity_relationships["projects_by_supplier"][supplier]:
                        self.entity_relationships["projects_by_supplier"][supplier].append(project)

                    # Record relationship: this project uses this supplier
                    if project not in self.entity_relationships["suppliers_by_project"]:
                        self.entity_relationships["suppliers_by_project"][project] = []
                    if supplier not in self.entity_relationships["suppliers_by_project"][project]:
                        self.entity_relationships["suppliers_by_project"][project].append(supplier)

        # Save query results for reference
        if query_result:
            self.last_query_result = query_result
        if sql_query:
            self.last_sql_query = sql_query

        # Track query complexity
        self.query_complexity_history.append(complexity)

        # Add to history with enhanced metadata
        self.conversation_history.append({
            "timestamp": timestamp,
            "question": question,
            "answer": answer,
            "entities": {k: v for k, v in entities.items() if k in ["project", "supplier"] and v is not None},
            "complexity": complexity,
            "sql_query": sql_query
        })

        # Maintain max history size
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
            self.query_complexity_history = self.query_complexity_history[-self.max_history:]

    def get_context_for_llm(self) -> str:
        """Format conversation history for the LLM with enhanced context"""
        if not self.conversation_history:
            return ""

        context = "Previous conversation:\n"
        for i, interaction in enumerate(self.conversation_history):
            context += f"User: {interaction['question']}\n"
            context += f"Assistant: {interaction['answer']}\n"
            if 'sql_query' in interaction and interaction['sql_query']:
                context += f"SQL used: {interaction['sql_query']}\n"

        # Add entity relationships context if available
        if self.entity_relationships:
            context += "\nEntity relationships:\n"

            if "suppliers_by_project" in self.entity_relationships:
                for project, suppliers in self.entity_relationships["suppliers_by_project"].items():
                    if suppliers:  # Only add if there are suppliers
                        context += f"Project '{project}' uses suppliers: {', '.join(suppliers)}\n"

            if "projects_by_supplier" in self.entity_relationships:
                for supplier, projects in self.entity_relationships["projects_by_supplier"].items():
                    if projects:  # Only add if there are projects
                        context += f"Supplier '{supplier}' works on projects: {', '.join(projects)}\n"

        return context

    def get_active_entities(self) -> Dict:
        """Get detected entities from the conversation"""
        return self.detected_entities

    def get_entity_relationships(self) -> Dict:
        """Get relationships between entities"""
        return self.entity_relationships

    def get_complexity_trend(self) -> int:
        """Analyze the trend of query complexity"""
        if not self.query_complexity_history:
            return 0

        # Return weighted average with more weight on recent queries
        weights = [1] * len(self.query_complexity_history)
        for i in range(len(weights)):
            weights[i] = (i + 1) ** 2  # Square the position to give more weight to recent

        weighted_sum = sum(c * w for c, w in zip(self.query_complexity_history, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0

    def should_offer_clarification(self) -> bool:
        """Determine if we should offer clarification based on history"""
        return self.clarification_needed

    def set_clarification_needed(self, needed: bool, suggestion: str = None):
        """Set the clarification status and suggestion"""
        self.clarification_needed = needed
        self.suggested_clarification = suggestion

    def reset(self):
        """Reset the conversation history"""
        self.conversation_history = []
        self.detected_entities = {}
        self.entity_relationships = {}
        self.last_query_result = None
        self.last_sql_query = None
        self.last_question = None
        self.query_complexity_history = []
        self.clarification_needed = False
        self.suggested_clarification = None
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

def analyze_query_complexity(question: str) -> dict:
    """
    Analyze the complexity of a query to determine if it needs special handling
    Returns a dictionary with complexity metrics and suggested handling approach
    """
    complexity_score = 0
    complexity_factors = []
    question_lower = question.lower()

    # Check for analytical operations
    analytical_terms = ["average", "mean", "median", "mode", "min", "max", "sum", "total", 
                      "count", "how many", "percentage", "ratio", "compare", "trend", 
                      "difference", "highest", "lowest", "largest", "smallest"]

    analytical_count = sum(1 for term in analytical_terms if term in question_lower)
    if analytical_count >= 2:
        complexity_score += 2
        complexity_factors.append("Multiple analytical operations")
    elif analytical_count == 1:
        complexity_score += 1
        complexity_factors.append("Analytical operation")

    # Check for multiple entities
    entity_terms = ["project", "supplier", "vendor", "item", "po", "purchase order", "term"]
    entity_count = sum(1 for term in entity_terms if term in question_lower)
    if entity_count >= 3:
        complexity_score += 2
        complexity_factors.append("Multiple entity types")
    elif entity_count == 2:
        complexity_score += 1
        complexity_factors.append("Two entity types")

    # Check for time-based queries
    time_terms = ["before", "after", "between", "during", "date", "month", "year", "time", "period"]
    if any(term in question_lower for term in time_terms):
        complexity_score += 1
        complexity_factors.append("Time-based query")

    # Check for complex conditions
    condition_terms = ["if", "when", "where", "unless", "except", "but", "only", "excluding"]
    condition_count = sum(1 for term in condition_terms if term in question_lower)
    if condition_count >= 2:
        complexity_score += 2
        complexity_factors.append("Multiple conditions")
    elif condition_count == 1:
        complexity_score += 1
        complexity_factors.append("Conditional query")

    # Check for multi-part questions
    question_marks = question.count('?')
    if question_marks > 1:
        complexity_score += question_marks
        complexity_factors.append("Multi-part question")

    # Check for sorting or grouping
    if any(term in question_lower for term in ["group", "sort", "order", "rank", "top", "bottom"]):
        complexity_score += 1
        complexity_factors.append("Sorting or grouping")

    # Determine the complexity level
    complexity_level = QUERY_COMPLEXITY["SIMPLE"]
    if complexity_score >= 5:
        complexity_level = QUERY_COMPLEXITY["VERY_COMPLEX"]
    elif complexity_score >= 3:
        complexity_level = QUERY_COMPLEXITY["COMPLEX"]
    elif complexity_score >= 1:
        complexity_level = QUERY_COMPLEXITY["MODERATE"]

    # Determine if query should be decomposed
    needs_decomposition = complexity_score >= 4 or question_marks > 1

    # Check if clarification might be needed
    needs_clarification = False
    clarification_suggestion = None

    if "which" in question_lower and not any(entity in question_lower for entity in ["project", "supplier", "vendor"]):
        needs_clarification = True
        clarification_suggestion = "Could you specify which project or supplier you're asking about?"

    if "when" in question_lower and not any(time_term in question_lower for time_term in ["date", "month", "year", "time", "period"]):
        needs_clarification = True
        clarification_suggestion = "Could you specify the time period you're interested in?"

    # Combine into result
    return {
        "complexity_score": complexity_score,
        "complexity_level": complexity_level,
        "complexity_factors": complexity_factors,
        "needs_decomposition": needs_decomposition,
        "needs_clarification": needs_clarification,
        "clarification_suggestion": clarification_suggestion
    }

def decompose_complex_question(question: str) -> List[dict]:
    """
    Break down complex multi-part questions into simpler sub-questions
    """
    # First check if the question actually needs decomposition
    complexity_analysis = analyze_query_complexity(question)
    if not complexity_analysis["needs_decomposition"]:
        return [{"question": question, "is_original": True}]

    sub_questions = []

    # Try to use LLM for decomposition
    try:
        system_message = "You are a question decomposition system. Your task is to break down complex questions into simpler sub-questions that can be answered independently."
        system_message += "\n\nAnalyze the complex question and break it down into 2-4 simpler questions that together would answer the original question."
        system_message += "\n\nReturn your answer in JSON format with the following structure:"
        system_message += "\n{\"sub_questions\": [\"simpler question 1\", \"simpler question 2\", ...]}"
        system_message += "\n\nExample: 'What are the top 3 most expensive items for Ring Road project and who are the suppliers?'"
        system_message += "\nResponse: {\"sub_questions\": [\"What are the top 3 most expensive items for Ring Road project?\", \"Who are the suppliers for the top 3 most expensive items in Ring Road project?\"]}"

        # Create message array with system instructions
        messages = [{"role": "system", "content": system_message}]

        # Add the current question
        messages.append({"role": "user", "content": f"Decompose this complex question: {question}"})

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
        json_match = re.search(r'\{[^{}]*\}', llm_response)
        if json_match:
            json_str = json_match.group(0)
            try:
                decomposition = json.loads(json_str)
                if "sub_questions" in decomposition and isinstance(decomposition["sub_questions"], list):
                    for sub_q in decomposition["sub_questions"]:
                        sub_questions.append({"question": sub_q, "is_original": False})
                    return sub_questions
            except json.JSONDecodeError:
                print(f"Failed to parse JSON from LLM response: {json_str}")
    except Exception as e:
        print(f"Error in question decomposition: {str(e)}")

    # Fallback: Simple rule-based decomposition if LLM failed
    if not sub_questions:
        # Split by question marks
        question_parts = re.split(r'\?', question)
        question_parts = [part.strip() + "?" for part in question_parts if part.strip()]

        # Remove the last element if it's just a question mark
        if question_parts and question_parts[-1] == "?":
            question_parts = question_parts[:-1]

        if len(question_parts) > 1:
            # We have multiple questions
            for part in question_parts:
                sub_questions.append({"question": part, "is_original": False})
        else:
            # Try to split by conjunctions
            conjunction_splits = re.split(r'\s+and\s+|\s+or\s+|\s+then\s+|\s+also\s+', question)
            if len(conjunction_splits) > 1:
                for split in conjunction_splits:
                    # Make sure it's a proper question
                    if not split.endswith('?'):
                        split += "?"
                    sub_questions.append({"question": split, "is_original": False})

    # If decomposition failed or returned no results, use the original question
    if not sub_questions:
        sub_questions.append({"question": question, "is_original": True})

    return sub_questions

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

def generate_clarification_questions(question: str, detected_entities: dict) -> List[str]:
    """
    Generate clarification questions when query intent is ambiguous
    """
    clarification_questions = []

    # Check for ambiguity in the question
    question_lower = question.lower()

    # If no entities detected, suggest clarifications based on entity types
    if not detected_entities.get("project") and not detected_entities.get("supplier"):
        # Check if question seems to reference a specific project but none detected
        if any(term in question_lower for term in ["project", "projects"]):
            clarification_questions.append("Which specific project are you asking about?")

        # Check if question seems to reference a specific supplier but none detected
        if any(term in question_lower for term in ["supplier", "vendor", "company"]):
            clarification_questions.append("Which specific supplier/vendor are you interested in?")

    # Check for ambiguous time references
    if any(term in question_lower for term in ["recent", "latest", "newest", "last", "previous"]):
        if not any(specific_time in question_lower for specific_time in ["year", "month", "week", "day", "date"]):
            clarification_questions.append("What time period are you referring to? (e.g., last month, last year)")

    # Check for ambiguous quantity references
    if any(term in question_lower for term in ["most", "top", "best", "highest", "lowest"]):
        if not any(specific_num in question_lower for specific_num in ["1", "2", "3", "4", "5", "one", "two", "three"]):
            clarification_questions.append("How many results would you like to see? (e.g., top 3, top 5)")

    # Check for ambiguous comparison references
    if any(term in question_lower for term in ["compare", "comparison", "versus", "vs"]):
        clarification_questions.append("What specific aspects would you like to compare?")

    # Use LLM to generate more sophisticated clarifications if needed
    if not clarification_questions and (not detected_entities.get("project") or not detected_entities.get("supplier")):
        try:
            system_message = "You are a clarification question generator. Your task is to identify ambiguities in user questions and suggest clarifying questions."
            system_message += "\n\nAnalyze the user's question about purchase orders and suggest ONE specific clarification question that would help provide a more precise answer."
            system_message += "\n\nIf the question is already clear and specific, respond with: \"NO_CLARIFICATION_NEEDED\""

            # Create message array with system instructions
            messages = [{"role": "system", "content": system_message}]

            # Add the current question with context
            context = f"User's question: {question}\n\n"
            if detected_entities.get("project"):
                context += f"Detected project: {detected_entities['project']}\n"
            if detected_entities.get("supplier"):
                context += f"Detected supplier: {detected_entities['supplier']}\n"

            messages.append({"role": "user", "content": context + "Generate one clarification question if needed."})

            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }

            resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            response = resp.json()
            suggestion = response['choices'][0]['message']['content'].strip()

            if suggestion and "NO_CLARIFICATION_NEEDED" not in suggestion:
                # Clean up the suggestion
                suggestion = suggestion.replace("Clarification question: ", "")
                suggestion = suggestion.strip('"')
                if not suggestion.endswith('?'):
                    suggestion += "?"
                clarification_questions.append(suggestion)

        except Exception as e:
            print(f"Error generating clarification questions: {str(e)}")

    return clarification_questions

def refine_context_from_history(question: str, conversation_history: list) -> dict:
    """
    Extract more nuanced context from conversation history
    Better handles pronoun references, implicit entity references, etc.
    """
    if not conversation_history:
        return {}

    context = {}
    question_lower = question.lower()

    # Check for pronouns and implicit references
    has_pronoun = any(pronoun in question_lower for pronoun in [
        " it ", " its ", " this ", " that ", " these ", " those ", 
        " them ", " they ", " their ", " he ", " she ", " him ", " her "])

    if has_pronoun:
        # Look at the most recent conversation entries for context
        for entry in reversed(conversation_history):
            if 'entities' in entry and entry['entities']:
                # Add any entities from the recent conversation
                for entity_type, entity_value in entry['entities'].items():
                    if entity_type not in context and entity_value:
                        context[entity_type] = entity_value

    # Check for follow-up indicators without pronouns
    follow_up_indicators = [
        "what about", "how about", "any other", "other ones", "more details", "tell me more",
        "could you elaborate", "can you explain", "and", "also", "as well", "additionally"
    ]

    is_follow_up = any(indicator in question_lower for indicator in follow_up_indicators)

    if is_follow_up and conversation_history:
        # Look at the most recent conversation entry for context
        last_entry = conversation_history[-1]
        if 'entities' in last_entry and last_entry['entities']:
            # Add any entities from the last conversation
            for entity_type, entity_value in last_entry['entities'].items():
                if entity_type not in context and entity_value:
                    context[entity_type] = entity_value

    return context

def process_question(question: str, use_history: bool = True) -> Tuple[str, str, List[str], List[List], Dict]:
    """
    Process the question and return answer, query, columns, results, and detected entities
    with enhanced error handling and query decomposition for complex questions
    """
    start_time = time.time()

    # Update system stats
    SYSTEM_STATS["queries_processed"] += 1

    # Analyze complexity first
    complexity_analysis = analyze_query_complexity(question)
    complexity_level = complexity_analysis["complexity_level"]

    if complexity_level >= QUERY_COMPLEXITY["COMPLEX"]:
        SYSTEM_STATS["complex_queries"] += 1

    # Check if the question needs clarification
    if complexity_analysis["needs_clarification"]:
        CONVERSATION.set_clarification_needed(True, complexity_analysis["clarification_suggestion"])
        clarification = complexity_analysis["clarification_suggestion"]
        return (f"I need some clarification to better answer your question: {clarification}", 
                "", ["Clarification Needed"], [[clarification]], {})

    # Clear any previous clarification flag
    CONVERSATION.set_clarification_needed(False)

    # Check if the question needs to be decomposed
    sub_questions = []
    if complexity_level >= QUERY_COMPLEXITY["COMPLEX"]:
        sub_questions = decompose_complex_question(question)
    else:
        sub_questions = [{"question": question, "is_original": True}]

    # If we have multiple sub-questions, process them one by one
    if len(sub_questions) > 1:
        combined_answer = f"Your question has multiple parts. Let me address each part:\n\n"
        all_results = []
        all_columns = []
        combined_entities = {}
        final_sql = ""

        for i, sub_q in enumerate(sub_questions):
            sub_answer, sub_sql, sub_columns, sub_results, sub_entities = process_single_question(
                sub_q["question"], use_history
            )

            # Append each part to the combined answer
            combined_answer += f"**Part {i+1}: {sub_q['question']}**\n{sub_answer}\n\n"

            # Accumulate entities
            for entity_type, entity_value in sub_entities.items():
                if entity_type not in combined_entities or not combined_entities[entity_type]:
                    combined_entities[entity_type] = entity_value

            # Keep track of results and SQL for the UI
            if sub_columns and sub_columns[0] != "Error" and sub_columns[0] != "Clarification Needed":
                if not all_columns:
                    all_columns = sub_columns
                    all_results = sub_results
                final_sql += f"-- Query for part {i+1}:\n{sub_sql}\n\n"

        # Update conversation with the combined information
        if use_history:
            CONVERSATION.add_interaction(
                question=question,
                answer=combined_answer,
                entities=combined_entities,
                sql_query=final_sql,
                query_result={"columns": all_columns, "results": all_results},
                complexity=complexity_level
            )

        query_time = time.time() - start_time
        SYSTEM_STATS["total_query_time"] += query_time
        SYSTEM_STATS["query_time_avg"] = SYSTEM_STATS["total_query_time"] / SYSTEM_STATS["queries_processed"]

        return combined_answer, final_sql, all_columns, all_results, combined_entities

    # Otherwise, process the single question
    result = process_single_question(question, use_history)

    query_time = time.time() - start_time
    SYSTEM_STATS["total_query_time"] += query_time
    SYSTEM_STATS["query_time_avg"] = SYSTEM_STATS["total_query_time"] / SYSTEM_STATS["queries_processed"]

    return result

def process_single_question(question: str, use_history: bool = True) -> Tuple[str, str, List[str], List[List], Dict]:
    """
    Process a single question (not decomposed) and return the results
    """
    try:
        detected_entities = {
            "project": None,
            "project_confidence": 0.0,
            "project_alternatives": [],
            "supplier": None,
            "supplier_confidence": 0.0,
            "supplier_alternatives": []
        }

        # First try LLM-based extraction with conversation history context
        llm_entities = detect_entities_with_llm(question, use_history=use_history)

        if llm_entities is not None:
            # Process project if LLM detected one
            if "project" in llm_entities and llm_entities["project"] is not None:
                project_candidate = llm_entities["project"]

                # Exact match
                if project_candidate in UNIQUE_PROJECTS:
                    detected_entities["project"] = project_candidate
                    detected_entities["project_confidence"] = 1.0

                    # Get alternative matches for reference
                    other_matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                    alternatives = [m for m in other_matches if m != project_candidate][:2]
                    detected_entities["project_alternatives"] = alternatives

                # Fuzzy match
                else:
                    matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                    if matches:
                        detected_entities["project"] = matches[0]
                        detected_entities["project_confidence"] = difflib.SequenceMatcher(None, project_candidate, matches[0]).ratio()
                        detected_entities["project_alternatives"] = matches[1:3] if len(matches) > 1 else []

            # Process supplier if LLM detected one
            if "supplier" in llm_entities and llm_entities["supplier"] is not None:
                supplier_candidate = llm_entities["supplier"]

                # Exact match
                if supplier_candidate in UNIQUE_SUPPLIERS:
                    detected_entities["supplier"] = supplier_candidate
                    detected_entities["supplier_confidence"] = 1.0

                    # Get alternative matches for reference
                    other_matches = difflib.get_close_matches(supplier_candidate, UNIQUE_SUPPLIERS, n=3, cutoff=0.5)
                    alternatives = [m for m in other_matches if m != supplier_candidate][:2]
                    detected_entities["supplier_alternatives"] = alternatives

                # Fuzzy match
                else:
                    matches = difflib.get_close_matches(supplier_candidate, UNIQUE_SUPPLIERS, n=3, cutoff=0.5)
                    if matches:
                        detected_entities["supplier"] = matches[0]
                        detected_entities["supplier_confidence"] = difflib.SequenceMatcher(None, supplier_candidate, matches[0]).ratio()
                        detected_entities["supplier_alternatives"] = matches[1:3] if len(matches) > 1 else []

        # Fallback to traditional entity extraction only if LLM response was invalid
        if llm_entities is None:
            project, project_confidence = extract_entity_from_question(question, "project", UNIQUE_PROJECTS)
            supplier, supplier_confidence = extract_entity_from_question(question, "supplier", UNIQUE_SUPPLIERS)

            detected_entities["project"] = project
            detected_entities["project_confidence"] = project_confidence
            detected_entities["supplier"] = supplier
            detected_entities["supplier_confidence"] = supplier_confidence

        # If we're using conversation history and no entities were detected in this question,
        # check if we have active entities from previous questions or can extract from context
        if use_history and (detected_entities["project"] is None or detected_entities["supplier"] is None):
            # First try the active entities directly
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

            # If still no entities, try more advanced context refinement
            if (detected_entities["project"] is None or detected_entities["supplier"] is None) and conversation_has_history:
                refined_context = refine_context_from_history(question, CONVERSATION.conversation_history)

                if detected_entities["project"] is None and "project" in refined_context:
                    detected_entities["project"] = refined_context["project"]
                    detected_entities["project_confidence"] = 0.7  # Even lower confidence for refined context

                if detected_entities["supplier"] is None and "supplier" in refined_context:
                    detected_entities["supplier"] = refined_context["supplier"]
                    detected_entities["supplier_confidence"] = 0.7  # Even lower confidence for refined context

        # If we still don't have entities, check if we need clarification
        if (detected_entities["project"] is None and detected_entities["supplier"] is None):
            clarification_questions = generate_clarification_questions(question, detected_entities)
            if clarification_questions:
                # Only use the first clarification question
                clarification = clarification_questions[0]
                CONVERSATION.set_clarification_needed(True, clarification)
                return (f"I need some clarification to better answer your question: {clarification}", 
                        "", ["Clarification Needed"], [[clarification]], detected_entities)

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

        # Optimize query if needed
        sql_query = optimize_query_for_performance(sql_query)

        # Execute query
        try:
            # Execute query
            columns, results = execute_query(sql_query)
            query_result = {"columns": columns, "results": results}

            if columns[0] == "Error":
                # If there was an error, try to recover
                fixed_query = recover_from_query_error(results[0][0], sql_query, question)
                if fixed_query and fixed_query != sql_query:
                    print(f"Attempting recovery with fixed query: {fixed_query}")
                    columns, results = execute_query(fixed_query)
                    query_result = {"columns": columns, "results": results}
                    if columns[0] != "Error":
                        sql_query = fixed_query  # Use the fixed query since it worked
                    else:
                        # If recovery failed, revert to original error
                        SYSTEM_STATS["failed_queries"] += 1
                        answer = f"I encountered an error with your query: {results[0][0]}"
                else:
                    # No recovery possible
                    SYSTEM_STATS["failed_queries"] += 1
                    answer = f"I encountered an error: {results[0][0]}"
            else:
                # Generate natural language answer with conversation context
                answer = generate_natural_language_answer(question, sql_query, columns, results, use_history)

            # Update conversation history if using it
            if use_history:
                # Analyze complexity for tracking
                complexity_analysis = analyze_query_complexity(question)

                CONVERSATION.add_interaction(
                    question=question,
                    answer=answer,
                    entities=detected_entities,
                    sql_query=sql_query,
                    query_result=query_result,
                    complexity=complexity_analysis["complexity_level"]
                )

            return answer, sql_query, columns, results, detected_entities

        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            SYSTEM_STATS["failed_queries"] += 1
            return error_message, f"Error: {str(e)}", ["Error"], [[error_message]], detected_entities

    except Exception as e:
        error_message = f"Unexpected error processing question: {str(e)}"
        SYSTEM_STATS["failed_queries"] += 1
        return error_message, f"Error: {str(e)}", ["Error"], [[error_message]], {}

def optimize_query_for_performance(sql_query: str) -> str:
    """
    Optimize SQL queries for better performance
    """
    # Convert to uppercase for consistent analysis
    query_upper = sql_query.upper()

    # Check if it's a SELECT query
    if not query_upper.startswith("SELECT"):
        return sql_query  # Don't optimize non-SELECT queries

    # Optimization 1: Add LIMIT if not present for large result sets
    if "LIMIT" not in query_upper and "ORDER BY" in query_upper:
        # Only add LIMIT to queries that are already sorting (likely want top N)
        if not re.search(r'\bLIMIT\s+\d+', query_upper):
            sql_query += " LIMIT 100"  # Add reasonable limit

    # Optimization 2: Add indexes hint if doing heavy filtering
    if "WHERE" in query_upper and ("PROJECT_NAME" in sql_query or "VENDOR_NAME" in sql_query):
        # Add comment to help PostgreSQL optimizer (won't affect query functionality)
        if not sql_query.startswith("-- Using indexes"):
            sql_query = f"-- Using indexes for PROJECT_NAME and VENDOR_NAME\n{sql_query}"

    # Optimization 3: Use COUNT(*) instead of COUNT(column) when possible
    sql_query = re.sub(r'COUNT\(\s*"[^"]+"\s*\)', 'COUNT(*)', sql_query, flags=re.IGNORECASE)

    return sql_query

def recover_from_query_error(error: str, original_query: str, question: str) -> str:
    """
    Attempt to recover from SQL errors by fixing the query
    """
    if not error or not original_query:
        return None

    # Common error patterns and fixes
    error_lower = error.lower()

    # Fix 1: Column doesn't exist
    column_not_exist_match = re.search(r'column "([^"]+)" does not exist', error)
    if column_not_exist_match:
        bad_column = column_not_exist_match.group(1)

        # Try to find a similar column name (case issues)
        for col_key, col_quoted in COLUMN_MAP.items():
            if bad_column.lower() == col_key.lower() or bad_column.upper() == col_key.upper():
                # Replace the bad column with the properly quoted one
                return original_query.replace(f'"{bad_column}"', col_quoted).replace(bad_column, col_quoted.replace('"', ''))

    # Fix 2: Invalid date format
    if "invalid input syntax for type date" in error_lower:
        # Try to fix date format issues
        date_pattern = r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}'
        dates_in_query = re.findall(date_pattern, original_query)

        if dates_in_query:
            fixed_query = original_query
            for date_str in dates_in_query:
                # Convert to PostgreSQL date format (YYYY-MM-DD)
                try:
                    if '/' in date_str:
                        parts = date_str.split('/')
                    else:
                        parts = date_str.split('-')

                    if len(parts) == 3:
                        # Assume MM/DD/YYYY or DD/MM/YYYY format
                        if len(parts[2]) == 4:  # YYYY is in position 2
                            fixed_date = f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
                        else:  # YYYY is in position 0 or needs to be expanded
                            year = parts[2]
                            if len(year) == 2:
                                year = f"20{year}" if int(year) < 50 else f"19{year}"
                            fixed_date = f"{year}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"

                        fixed_query = fixed_query.replace(date_str, fixed_date)
                except:
                    # If date parsing fails, continue to next fix attempt
                    pass

            return fixed_query

    # Fix 3: Syntax error in the query
    if "syntax error at or near" in error_lower:
        # Try to generate a completely new query
        try:
            system_message = "You are a SQL query repair expert. Fix the broken SQL query based on the error message."
            system_message += "\n\nOriginal question: " + question
            system_message += "\n\nBroken SQL query: " + original_query
            system_message += "\n\nError message: " + error
            system_message += "\n\nProvide ONLY the fixed SQL query with no explanations or comments."

            messages = [{"role": "system", "content": system_message}]
            messages.append({"role": "user", "content": "Fix the SQL query based on the error message."})  

            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }

            resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            response = resp.json()
            fixed_query = response['choices'][0]['message']['content'].strip()

            # Clean the response - remove any markdown code fences or explanation text
            fixed_query = fixed_query.replace('```sql', '').replace('```', '').strip()
            if fixed_query.startswith('`') and fixed_query.endswith('`'):
                fixed_query = fixed_query[1:-1].strip()

            # Extract just the SQL if there's explanation text
            if "SELECT" in fixed_query.upper():
                select_pos = fixed_query.upper().find("SELECT")
                if select_pos > 0:
                    fixed_query = fixed_query[select_pos:]

            return fixed_query

        except Exception as e:
            print(f"Error attempting to fix query: {str(e)}")
            return None

    # No fix was possible
    return None

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
        system_message += "\n\nFor complex analytical queries:\n"
        system_message += "1. Use appropriate aggregation functions (SUM, AVG, COUNT, etc.)\n"
        system_message += "2. Include appropriate GROUP BY clauses\n"
        system_message += "3. For top/bottom N queries, use ORDER BY with LIMIT\n"
        system_message += "4. For date ranges, use appropriate date functions and comparisons\n"

        # Create message array with system instructions
        messages = [{"role": "system", "content": system_message}]

        # Add conversation history context if available and requested
        if use_history and CONVERSATION.conversation_history:
            context_message = "Previous conversation context:\n"

            # Add the last few interactions for context
            for interaction in CONVERSATION.conversation_history[-3:]:  # Use last 3 interactions for more context
                context_message += f"User asked: {interaction['question']}\n"
                if 'sql_query' in interaction and interaction['sql_query']:
                    context_message += f"SQL used: {interaction['sql_query']}\n"

            # Add currently active entities from the conversation
            active_entities = CONVERSATION.get_active_entities()
            if active_entities:
                context_message += "\nActive entities from conversation:\n"
                for entity_type, entity_value in active_entities.items():
                    context_message += f"{entity_type}: {entity_value}\n"

            # Add entity relationships for better context awareness
            entity_relationships = CONVERSATION.get_entity_relationships()
            if entity_relationships:
                context_message += "\nKnown entity relationships:\n"

                if "suppliers_by_project" in entity_relationships:
                    for project, suppliers in entity_relationships["suppliers_by_project"].items():
                        if suppliers:  # Only add if there are suppliers
                            context_message += f"Project '{project}' uses suppliers: {', '.join(suppliers)}\n"

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

        # Validate the query syntax before returning
        if not sql_query.upper().startswith("SELECT") and not sql_query.upper().startswith("WITH"):
            # Query doesn't look valid, regenerate with more explicit instructions
            messages[-1]["content"] = f"Generate ONLY a valid PostgreSQL SELECT query to answer this question: {question}"

            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False
            }

            resp = requests.post(f"{OLLAMA_HOST}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            response = resp.json()
            sql_query = response['choices'][0]['message']['content'].strip()

            # Clean the SQL query again
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            if sql_query.startswith('`') and sql_query.endswith('`'):
                sql_query = sql_query[1:-1].strip()

        return sql_query
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

def determine_optimal_response_format(question: str, columns: list, results: list) -> str:
    """
    Determine the best way to present results (table, summary, chart suggestion)
    """
    if not results or len(results) == 0:
        return "empty"

    question_lower = question.lower()
    result_count = len(results)
    column_count = len(columns) if columns else 0

    # Check for analytical questions that benefit from summaries
    analytical_terms = ["average", "mean", "median", "min", "max", "sum", "total", 
                      "count", "how many", "percentage", "ratio", "compare", "trend", 
                      "difference", "highest", "lowest"]

    is_analytical = any(term in question_lower for term in analytical_terms)

    # Check for questions asking for specific items or entities
    specific_entity_terms = ["which", "what", "who", "show me", "list", "find"]
    is_entity_lookup = any(term in question_lower for term in specific_entity_terms)

    # Check if results are too large for direct display
    is_large_result = result_count > 10
    is_wide_result = column_count > 5

    # Determine format based on query characteristics
    if is_analytical and result_count == 1 and column_count <= 3:
        return "summary"  # Single value or small analytical result
    elif is_entity_lookup and not is_large_result and not is_wide_result:
        return "table"  # Clean table presentation for moderate result sets
    elif is_large_result or is_wide_result:
        return "summary_with_examples"  # Summarize large results with examples
    else:
        return "table"  # Default to table format

def generate_natural_language_answer(question: str, sql_query: str, columns: list, results: list, use_history: bool = True) -> str:
    if not results or len(results) == 0:
        return f"There are no results for this query. Please try adjusting your question or criteria."

    # Determine the optimal response format
    response_format = determine_optimal_response_format(question, columns, results)

    # Format results as readable string based on optimal format
    if columns[0] == "Error":
        result_str = results[0][0] if results and len(results) > 0 else "Unknown error"
    else:
        if response_format == "table" or response_format == "empty":
            header = " | ".join(columns)
            sep = " | ".join(["---"] * len(columns))
            rows = [" | ".join(str(cell) for cell in row) for row in results]
            result_str = f"{header}\n{sep}\n" + "\n".join(rows)
        elif response_format == "summary_with_examples":
            result_str = f"Query returned {len(results)} results with {len(columns)} columns.\n\n"
            result_str += "Sample results (first 3 rows):\n"
            header = " | ".join(columns)
            sep = " | ".join(["---"] * len(columns))
            sample_rows = [" | ".join(str(cell) for cell in row) for row in results[:3]]
            result_str += f"{header}\n{sep}\n" + "\n".join(sample_rows)
        else:  # summary format
            result_str = f"Result: {len(results)} rows with columns: {', '.join(columns)}\n"
            # Add first few results
            for i, row in enumerate(results[:3]):
                result_str += f"Row {i+1}: " + ", ".join(f"{col}: {val}" for col, val in zip(columns, row)) + "\n"

    # Build the prompt using standard string concatenation to avoid any f-string issues
    prompt = "Given the user's question and the SQL query result below, write a clear, concise answer in English. If the result is empty, say so.\n\n"
    prompt += f"User question: {question}\n\n"
    prompt += f"SQL query: {sql_query}\n\n"
    prompt += f"Query result:\n{result_str}\n\n"
    prompt += "Answer in clear, simple English. Be conversational. Don't repeat all values unless requested. "
    prompt += "For analytical questions, include insights about the data. "
    prompt += "For empty results, suggest ways to broaden the search."

    if use_history and CONVERSATION.conversation_history:
        prompt += "\n\nThis is part of an ongoing conversation. Previous context:\n"
        # Add up to the last 2 conversation exchanges for context
        for interaction in CONVERSATION.conversation_history[-2:]:
            prompt += f"User: {interaction['question']}\n"
            prompt += f"Assistant: {interaction['answer']}\n"

        # Remind the model to refer to previously mentioned entities if relevant
        prompt += "\nIf this is a follow-up question, refer to entities mentioned in previous questions appropriately. "
        prompt += "For example, if a project was mentioned before but not in this question, still reference it by name."

        # If we have entity relationships, include them for better context
        entity_relationships = CONVERSATION.get_entity_relationships()
        if entity_relationships:
            prompt += "\n\nKnown entity relationships:\n"

            if "suppliers_by_project" in entity_relationships and detected_entities.get("project"):
                project = detected_entities["project"]
                if project in entity_relationships["suppliers_by_project"]:
                    suppliers = entity_relationships["suppliers_by_project"][project]
                    if suppliers:
                        prompt += f"Project '{project}' uses suppliers: {', '.join(suppliers)}\n"

            if "projects_by_supplier" in entity_relationships and detected_entities.get("supplier"):
                supplier = detected_entities["supplier"]
                if supplier in entity_relationships["projects_by_supplier"]:
                    projects = entity_relationships["projects_by_supplier"][supplier]
                    if projects:
                        prompt += f"Supplier '{supplier}' works on projects: {', '.join(projects)}\n"

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
        if query.strip().upper().startswith("SELECT") or query.strip().upper().startswith("WITH"):
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
    with gr.Blocks(title="RME PO Query Assistant rev.17 (Enhanced PostgreSQL with Memory)") as interface:
        # Title and version info
        gr.Markdown(f"# PO Follow-Up Query AI (rev.17, PostgreSQL with Advanced Query Understanding)")

        # Three column layout - Chat on left, Entities in middle, System Info on right
        with gr.Row():
            # Left column for chat interface
            with gr.Column(scale=5):
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

            # Middle column for entity detection
            with gr.Column(scale=3):
                detected_entities = gr.Markdown("### Detected Entities\nNo entities detected yet.", label="Current Entities")

                with gr.Row():
                    supplier_btn = gr.Button("Show Suppliers", elem_id="supplier-btn")
                    project_btn = gr.Button("Show Projects", elem_id="project-btn")

                # Visual representation of conversation complexity
                conversation_complexity = gr.Label(label="Conversation Complexity", value="Simple")

            # Right column for system information
            with gr.Column(scale=2):
                system_info = gr.Markdown(f"### System Stats\n" + 
                                        f"- Version: {SYSTEM_STATS['version']}\n" +
                                        f"- Queries: 0\n" +
                                        f"- Complex: 0\n" +
                                        f"- Failed: 0\n" +
                                        f"- Avg Time: 0.00s")

                # Add visualization of entity relationships (simplified)
                entity_relationships = gr.Markdown("### Entity Relationships\nNo relationships discovered yet.")

        # SQL and results section (below the conversation)
        with gr.Accordion("SQL Query and Results", open=False):
            query_output = gr.Code(label="Current SQL Query", language="sql")
            results = gr.Dataframe(label="Query Results")

        # Informational text moved to bottom of the interface
        gr.Markdown(
            "### System Information\n"
            "This enhanced AI assistant queries the merged PO table with improved question understanding.\n\n"
            "- Using local PostgreSQL database with **861,403 PO records**\n"
            "- Handles complex analytical questions with multi-part decomposition\n"
            "- Provides active clarification for ambiguous questions\n"
            "- Tracks relationships between entities for better context\n"
            "- Automatically recovers from common query errors\n"
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
            return [], "### Detected Entities\nMemory has been reset. The chatbot will no longer remember previous questions and entities.", "", None, "Simple", "### Entity Relationships\nNo relationships discovered yet.", update_system_info()

        def on_submit(question, chat_history):
            if not question.strip():
                return chat_history, question, "### Detected Entities\nNo query entered", "", None, "Simple", "### Entity Relationships\nNo relationships discovered yet.", system_info

            try:
                # Process the question with conversation history enabled
                answer, query, columns, results_data, entities = process_question(question, use_history=True)

                # Update the chat history with the new question and answer
                chat_history = chat_history + [(question, answer)]

                # Format detected entities for display in the right panel
                entities_markdown = "### Detected Entities\n"

                # Display project and alternatives if available
                if entities.get("project"):
                    entities_markdown += f"**Project:** {entities['project']} *(confidence: {entities.get('project_confidence', 0):.2f})*\n"

                    # Show alternative project matches if any exist
                    if entities.get("project_alternatives") and len(entities["project_alternatives"]) > 0:
                        entities_markdown += "**Alternative projects:** "
                        entities_markdown += ", ".join([f"`{alt}`" for alt in entities["project_alternatives"]])
                        entities_markdown += "\n"
                else:
                    entities_markdown += "**Project:** None\n"

                # Display supplier and alternatives if available
                if entities.get("supplier"):
                    entities_markdown += f"**Supplier:** {entities['supplier']} *(confidence: {entities.get('supplier_confidence', 0):.2f})*\n"

                    # Show alternative supplier matches if any exist
                    if entities.get("supplier_alternatives") and len(entities["supplier_alternatives"]) > 0:
                        entities_markdown += "**Alternative suppliers:** "
                        entities_markdown += ", ".join([f"`{alt}`" for alt in entities["supplier_alternatives"]])
                        entities_markdown += "\n"
                else:
                    entities_markdown += "**Supplier:** None\n"

                # Check if clarification was needed
                if CONVERSATION.should_offer_clarification():
                    entities_markdown += "\n**Clarification needed:** Yes\n"
                    if CONVERSATION.suggested_clarification:
                        entities_markdown += f"*Suggestion: {CONVERSATION.suggested_clarification}*\n"

                # Get conversation complexity
                complexity_trend = CONVERSATION.get_complexity_trend()
                complexity_label = "Simple"
                if complexity_trend >= QUERY_COMPLEXITY["VERY_COMPLEX"]:
                    complexity_label = "Very Complex"
                elif complexity_trend >= QUERY_COMPLEXITY["COMPLEX"]:
                    complexity_label = "Complex"
                elif complexity_trend >= QUERY_COMPLEXITY["MODERATE"]:
                    complexity_label = "Moderate"

                # Format entity relationships
                relationships_markdown = "### Entity Relationships\n"
                entity_relationships = CONVERSATION.get_entity_relationships()

                if entity_relationships and ("suppliers_by_project" in entity_relationships or "projects_by_supplier" in entity_relationships):
                    if "suppliers_by_project" in entity_relationships:
                        for project, suppliers in entity_relationships["suppliers_by_project"].items():
                            if suppliers:  # Only add if there are suppliers
                                relationships_markdown += f"**{project}** works with: {', '.join(suppliers)}\n"

                    if "projects_by_supplier" in entity_relationships:
                        for supplier, projects in entity_relationships["projects_by_supplier"].items():
                            if projects:  # Only add if there are projects
                                relationships_markdown += f"**{supplier}** supplies: {', '.join(projects)}\n"
                else:
                    relationships_markdown += "No entity relationships discovered yet.\n"

                if query.startswith("Error") or (columns and columns[0] == "Error"):
                    error_message = results_data[0][0] if columns[0] == "Error" else query
                    return chat_history, "", entities_markdown, query, gr.Dataframe(value=[[error_message]], headers=["Error"]), complexity_label, relationships_markdown, update_system_info()

                # Create dataframe for results
                df = gr.Dataframe(value=results_data, headers=columns)
                return chat_history, "", entities_markdown, query, df, complexity_label, relationships_markdown, update_system_info()

            except Exception as e:
                error_message = f"Unexpected error: {str(e)}"
                chat_history = chat_history + [(question, error_message)]
                return chat_history, "", "### Detected Entities\nError in entity detection", "", None, "Simple", "### Entity Relationships\nNo relationships discovered yet.", update_system_info()

        def update_system_info():
            return (f"### System Stats\n" + 
                   f"- Version: {SYSTEM_STATS['version']}\n" +
                   f"- Queries: {SYSTEM_STATS['queries_processed']}\n" +
                   f"- Complex: {SYSTEM_STATS['complex_queries']}\n" +
                   f"- Failed: {SYSTEM_STATS['failed_queries']}\n" +
                   f"- Avg Time: {SYSTEM_STATS['query_time_avg']:.2f}s")

        # Set up event listeners
        submit_btn.click(
            fn=on_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, detected_entities, query_output, results, conversation_complexity, entity_relationships, system_info]
        )

        # Add Enter key submission functionality
        question_input.submit(
            fn=on_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, detected_entities, query_output, results, conversation_complexity, entity_relationships, system_info]
        )

        # Connect the clear history button
        clear_history_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[chatbot, detected_entities, query_output, results, conversation_complexity, entity_relationships, system_info]
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
                "What are the top 3 most expensive items for Ring Road project and who are the suppliers?",  # Complex question
                "Compare the average item prices between DUBA PARK projects and NEOM projects",  # Analytical question
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

@app.get("/system-stats", response_class=HTMLResponse)
def system_stats():
    html = f"<h2>System Statistics</h2>"
    html += f"<p>Version: {SYSTEM_STATS['version']}</p>"
    html += f"<p>Queries Processed: {SYSTEM_STATS['queries_processed']}</p>"
    html += f"<p>Complex Queries: {SYSTEM_STATS['complex_queries']}</p>"
    html += f"<p>Failed Queries: {SYSTEM_STATS['failed_queries']}</p>"
    html += f"<p>Average Query Time: {SYSTEM_STATS['query_time_avg']:.2f}s</p>"
    return HTMLResponse(content=html)

if __name__ == "__main__":
    initialize_unique_lists()
    interface = create_interface()
    app = gr.mount_gradio_app(app, interface, path="/")
    # Use port 7870 for rev.17 to avoid conflict with rev.16 on port 7869
    webbrowser.open("http://localhost:7870")
    uvicorn.run(app, host="0.0.0.0", port=7870)
