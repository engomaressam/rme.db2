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

# 3. Fix: Add a simple try-except block to the process_question function
process_question_pattern = r'def process_question\(message, history, doc_paths_state\):'
process_question_replacement = """def process_question(message, history, doc_paths_state):
    # Add try-except to catch and display errors
    try:"""

content = re.sub(process_question_pattern, process_question_replacement, content)

# 4. Find the return statement in process_question and add except block after it
return_pattern = r'return history, doc_paths, doc_buttons_html'
return_replacement = """return history, doc_paths, doc_buttons_html
    except Exception as e:
        logger.exception(f"Exception in process_question: {e}")
        error_message = f"Error processing your question: {str(e)}"
        history.append((message, error_message))
        return history, doc_paths_state, ""
"""

content = re.sub(return_pattern, return_replacement, content)

# 5. Add more logging to the generate_response function
generate_response_pattern = r'def generate_response\(prompt, system_prompt=None\):'
generate_response_replacement = """def generate_response(prompt, system_prompt=None):
    # Add more detailed logging
    logger.info("Starting generate_response function")"""

content = re.sub(generate_response_pattern, generate_response_replacement, content)

# Write the modified content to the output file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Original file backed up to: {BACKUP_FILE}")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
