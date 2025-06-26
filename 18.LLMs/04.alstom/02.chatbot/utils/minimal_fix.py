"""
Minimal fix for the Alstom chatbot display issue
"""
import os
import shutil
import re

# Path to the original chatbot file
ORIGINAL_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.10.py"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.16.py"

# Create backup
print(f"Creating output file: {OUTPUT_FILE}")
shutil.copy2(ORIGINAL_FILE, OUTPUT_FILE)

# Read the file
with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Update the chatbot definition to use a simpler format
chatbot_pattern = r'chatbot = gr\.Chatbot\(\s*\[\],\s*elem_id="chatbot",\s*height=500,\s*type="messages"\s*\)'
chatbot_replacement = 'chatbot = gr.Chatbot([], elem_id="chatbot", height=500)'

content = re.sub(chatbot_pattern, chatbot_replacement, content)

# Fix 2: Update the submit event to use a simpler format
submit_pattern = r'submit_event = msg\.submit\(\s*process_question,\s*\[msg, chatbot, doc_paths\],\s*\[chatbot, doc_paths, doc_buttons\]\s*\)'
submit_replacement = 'submit_event = msg.submit(process_question, [msg, chatbot, doc_paths], [chatbot, doc_paths, doc_buttons], show_progress=True)'

content = re.sub(submit_pattern, submit_replacement, content)

# Fix 3: Update the submit click event to use a simpler format
click_pattern = r'submit_click_event = submit\.click\(\s*process_question,\s*\[msg, chatbot, doc_paths\],\s*\[chatbot, doc_paths, doc_buttons\]\s*\)'
click_replacement = 'submit_click_event = submit.click(process_question, [msg, chatbot, doc_paths], [chatbot, doc_paths, doc_buttons], show_progress=True)'

content = re.sub(click_pattern, click_replacement, content)

# Write the modified content back to the file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully applied minimal fixes to the chatbot")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
