"""
Targeted fix for the Gradio format error in the Alstom chatbot
Uses the latest version (rev.17) as the base to maintain API connection fixes
"""
import os
import shutil
import re

# Path to the latest chatbot file
LATEST_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.17.py"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.18.py"

# Create backup
print(f"Creating output file: {OUTPUT_FILE}")
shutil.copy2(LATEST_FILE, OUTPUT_FILE)

# Read the file
with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Update the chatbot definition to explicitly use the messages format
chatbot_pattern = r'chatbot = gr\.Chatbot\(.*?\)'
chatbot_replacement = 'chatbot = gr.Chatbot([], elem_id="chatbot", height=500, type="messages")'

content = re.sub(chatbot_pattern, chatbot_replacement, content)

# Fix 2: Find the process_question function and modify how it returns data
process_question_pattern = r'def process_question\(message, history, doc_paths_state\):'
process_question_start = content.find(process_question_pattern)

if process_question_start > 0:
    # Find the return statement
    return_pattern = r'return history, doc_paths, doc_buttons_html'
    return_pos = content.find(return_pattern, process_question_start)
    
    if return_pos > 0:
        # Insert code to format the history properly before the return statement
        format_code = """
        # Ensure history is in the correct format for Gradio
        formatted_history = []
        for item in history:
            if isinstance(item, tuple) and len(item) == 2:
                formatted_history.append(list(item))
            elif isinstance(item, list) and len(item) == 2:
                formatted_history.append(item)
            else:
                logger.warning(f"Skipping invalid history item: {item}")
        
        """
        
        # Replace the return statement
        modified_return = f"{format_code}\n        return formatted_history, doc_paths, doc_buttons_html"
        content = content[:return_pos] + modified_return + content[return_pos + len(return_pattern):]

# Fix 3: Update the history.append line to use lists instead of tuples
history_append_pattern = r'history\.append\(\(message, response\)\)'
history_append_replacement = 'history.append([message, response])'

content = re.sub(history_append_pattern, history_append_replacement, content)

# Write the modified content back to the file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully applied Gradio format fixes to the chatbot")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
