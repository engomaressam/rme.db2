"""
Fix for the Gradio format error in the Alstom chatbot
"""
import os
import shutil
import re

# Path to the original chatbot file
ORIGINAL_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.10.py"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.17.py"

# Create backup
print(f"Creating output file: {OUTPUT_FILE}")
shutil.copy2(ORIGINAL_FILE, OUTPUT_FILE)

# Read the file
with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Update the chatbot definition to use the correct format
chatbot_pattern = r'chatbot = gr\.Chatbot\(\s*\[\],\s*elem_id="chatbot",\s*height=500,\s*type="messages"\s*\)'
chatbot_replacement = 'chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=500)'

content = re.sub(chatbot_pattern, chatbot_replacement, content)

# Fix 2: Update the process_question function to ensure proper message format
process_question_pattern = r'# Update history and return\s*history\.append\(\(message, response\)\)'
process_question_replacement = """# Update history and return
        # Ensure proper message format for Gradio chatbot
        try:
            history.append((message, response))
        except Exception as e:
            logger.exception(f"Error appending to history: {e}")
            # Try a different format if the standard one fails
            if not history:
                history = []
            history.append([message, response])"""

content = re.sub(process_question_pattern, process_question_replacement, content)

# Write the modified content back to the file
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully applied format fixes to the chatbot")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
