"""
Fix syntax error in the chatbot file
"""
import os
import shutil

# Path to the chatbot file
CHATBOT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.12.py"
BACKUP_FILE = CHATBOT_FILE + ".bak"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.13.py"

# Create backup
print(f"Creating backup: {BACKUP_FILE}")
shutil.copy2(CHATBOT_FILE, BACKUP_FILE)

# Read the file
with open(CHATBOT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the syntax error with escaped newlines
fixed_content = content.replace('context_text += "\\n\\nRelevant information from documents:\\n"', 
                              'context_text += "\\n\\nRelevant information from documents:\\n"')

# Fix any other potential issues with escaped newlines
fixed_content = fixed_content.replace('prompt += "\\n\\nHere is relevant information from the project documents:"',
                                    'prompt += "\\n\\nHere is relevant information from the project documents:"')

# Write the updated content
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print(f"Original file backed up to: {BACKUP_FILE}")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
