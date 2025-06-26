"""
Fix the display issue in the Alstom Project Assistant chatbot
This script makes targeted changes to ensure responses are displayed in the web interface
"""
import os
import re
import shutil

# Path to the chatbot file
CHATBOT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.11.py"
BACKUP_FILE = CHATBOT_FILE + ".bak"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.14.py"

# Create backup
print(f"Creating backup: {BACKUP_FILE}")
shutil.copy2(CHATBOT_FILE, BACKUP_FILE)

# Read the file
with open(CHATBOT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix: Add debug logging to the submit event
submit_event_pattern = r'submit_event = msg\.submit\(\s+process_question,\s+\[msg, chatbot, doc_paths\],\s+\[chatbot, doc_paths, doc_buttons\]\s+\)'
submit_event_replacement = """submit_event = msg.submit(
                    process_question, 
                    [msg, chatbot, doc_paths], 
                    [chatbot, doc_paths, doc_buttons],
                    api_name="process_question"  # Add explicit API name
                )"""

content = re.sub(submit_event_pattern, submit_event_replacement, content)

# 2. Fix: Add debug logging to the submit click event
submit_click_pattern = r'submit_click_event = submit\.click\(\s+process_question,\s+\[msg, chatbot, doc_paths\],\s+\[chatbot, doc_paths, doc_buttons\]\s+\)'
submit_click_replacement = """submit_click_event = submit.click(
                    process_question, 
                    [msg, chatbot, doc_paths], 
                    [chatbot, doc_paths, doc_buttons],
                    api_name="process_question_click"  # Add explicit API name
                )"""

content = re.sub(submit_click_pattern, submit_click_replacement, content)

# 3. Fix: Simplify the process_question function to ensure responses are displayed
process_question_pattern = r'def process_question\(message, history, doc_paths_state\):.*?return history, doc_paths, doc_buttons_html'
process_question_replacement = """def process_question(message, history, doc_paths_state):
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
        system_prompt = "You are the Alstom Project Assistant, a helpful AI that provides accurate information about the Alstom project based on the provided documents. Keep responses brief and to the point. Focus only on information in the provided documents. If you're not sure, say so rather than making things up. Respond within 30 seconds."
        
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
        
        # Update history with the response
        history.append((message, response))
        
        # Return the updated history and other values
        return history, doc_paths, doc_buttons_html
    except Exception as e:
        logger.exception(f"Exception in process_question: {e}")
        error_message = f"Error processing your question: {str(e)}"
        history.append((message, error_message))
        return history, doc_paths_state, ""
"""

content = re.sub(process_question_pattern, process_question_replacement, content, flags=re.DOTALL)

# Write the modified content to the output file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Original file backed up to: {BACKUP_FILE}")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
