"""
Quick fix for the Alstom chatbot to ensure responses are displayed
"""
import os
import shutil

# Path to the original chatbot file
ORIGINAL_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.10.py"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.15.py"

# Create backup
print(f"Creating output file: {OUTPUT_FILE}")
shutil.copy2(ORIGINAL_FILE, OUTPUT_FILE)

# Read the file
with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the process_question function
process_question_start = -1
process_question_end = -1
for i, line in enumerate(lines):
    if "def process_question(message, history, doc_paths_state):" in line:
        process_question_start = i
    if process_question_start > 0 and "return history, doc_paths, doc_buttons_html" in line:
        process_question_end = i
        break

if process_question_start > 0 and process_question_end > 0:
    # Add try-except block
    lines[process_question_start] = "def process_question(message, history, doc_paths_state):\n    try:\n"
    
    # Add indentation to all lines in the function
    for i in range(process_question_start + 1, process_question_end + 1):
        lines[i] = "    " + lines[i]
    
    # Add except block after return statement
    lines[process_question_end] = lines[process_question_end] + """    except Exception as e:
        logger.exception(f"Exception in process_question: {e}")
        error_message = f"Error processing your question: {str(e)}"
        history.append((message, error_message))
        return history, doc_paths_state, ""
"""

    # Write the modified content back to the file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Successfully added try-except block to process_question function")
    print(f"Fixed file written to: {OUTPUT_FILE}")
    print("\nTo use the fixed chatbot, run:")
    print(f"python {OUTPUT_FILE}")
else:
    print("Could not find process_question function or return statement")
