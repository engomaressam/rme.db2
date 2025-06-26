"""
Fix the display issue in the chatbot
This script makes a targeted fix to the rev.11 version to ensure responses are displayed
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

# Find the process_question function
process_question_pattern = r'def process_question\(message, history, doc_paths_state\):'
match = re.search(process_question_pattern, content)

if match:
    # Find the position to insert the try-except block
    position = match.end()
    
    # Create the try-except block
    try_except_block = """
    # Wrap in try-except to catch and display any errors
    try:
"""

    # Insert the try-except block
    modified_content = content[:position] + try_except_block + content[position:]
    
    # Find the end of the process_question function to add the except block
    end_pattern = r'return history, doc_paths, doc_buttons_html'
    end_match = re.search(end_pattern, modified_content)
    
    if end_match:
        end_position = end_match.end()
        
        # Create the except block
        except_block = """
    except Exception as e:
        logger.exception(f"Exception in process_question: {e}")
        error_message = f"Error processing your question: {str(e)}"
        history.append((message, error_message))
        return history, doc_paths_state, ""
"""
        
        # Insert the except block
        modified_content = modified_content[:end_position] + except_block + modified_content[end_position:]
        
        # Write the modified content to the output file
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"Original file backed up to: {BACKUP_FILE}")
        print(f"Fixed file written to: {OUTPUT_FILE}")
        print("\nTo use the fixed chatbot, run:")
        print(f"python {OUTPUT_FILE}")
    else:
        print("Could not find the end of the process_question function")
else:
    print("Could not find the process_question function")
