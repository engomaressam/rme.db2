"""
Fix for the process_question function in the Alstom chatbot
"""
import os
import shutil
import re

# Path to the chatbot file
CHATBOT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.11.py"
BACKUP_FILE = CHATBOT_FILE + ".bak2"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.12.py"

# Create backup
print(f"Creating backup: {BACKUP_FILE}")
shutil.copy2(CHATBOT_FILE, BACKUP_FILE)

# Read the file
with open(CHATBOT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the process_question function
process_question_pattern = r'def process_question\(message, history, doc_paths_state\):.*?(?=def find_available_port|# Find an available port)'
process_question_replacement = '''def process_question(message, history, doc_paths_state):
    """Process a user question and generate a response"""
    try:
        logger.info(f"Processing question: {message}")
        
        if not message:
            return history, doc_paths_state, ""
        
        # Find relevant documents
        start_time = time.time()
        relevant_docs = find_relevant_documents(message)
        end_time = time.time()
        logger.info(f"Found {len(relevant_docs)} documents in {end_time - start_time:.2f} seconds")
        
        # Extract document paths
        doc_paths = []
        context_text = ""
        
        # Add document context if available
        if relevant_docs:
            context_text += "\\n\\nRelevant information from documents:\\n"
            
            for i, doc in enumerate(relevant_docs, 1):
                doc_path = doc["path"]
                doc_paths.append(doc_path)
                
                # Add document content to context
                context_text += f"\\n[Document {i}] {os.path.basename(doc_path)}\\n"
                context_text += f"{doc['content']}\\n"
        
        # Create system prompt
        system_prompt = "You are the Alstom Project Assistant, a helpful AI that provides accurate information about the Alstom project based on the provided documents. If you don't know the answer or if the information is not in the documents, say so."
        
        # Create user prompt with context
        prompt = message
        if context_text:
            prompt += "\\n\\nHere is relevant information from the project documents:" + context_text
        
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Generate response
        logger.info("Generating response...")
        response = generate_response(prompt, system_prompt)
        logger.info(f"Response received, length: {len(response)} characters")
        
        # Create document buttons HTML
        doc_buttons_html = ""
        if doc_paths:
            doc_buttons_html = "<div style='margin-top: 10px;'><p><strong>Relevant Documents:</strong></p>"
            for i, doc_path in enumerate(doc_paths, 1):
                doc_name = os.path.basename(doc_path)
                doc_url = f"http://{DOC_SERVER_HOST}:{DOC_SERVER_PORT}/document?path={urllib.parse.quote(doc_path)}"
                doc_buttons_html += f"<a href='{doc_url}' target='_blank' style='display: inline-block; margin: 5px; padding: 5px 10px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 4px;'>{i}. {doc_name}</a>"
            doc_buttons_html += "</div>"
        
        # Update history and return
        history.append((message, response))
        return history, doc_paths, doc_buttons_html
    except Exception as e:
        logger.exception(f"Exception in process_question: {e}")
        error_message = f"Error processing your question: {str(e)}"
        history.append((message, error_message))
        return history, doc_paths_state, ""

'''

# Replace the function
new_content = re.sub(process_question_pattern, process_question_replacement, content, flags=re.DOTALL)

# Write the updated content
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Original file backed up to: {BACKUP_FILE}")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
