"""
Simplified fix for the Alstom chatbot display issue
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
    content = f.read()

# Add a simple fix to ensure the response is displayed
# 1. Find the chatbot interface definition
chatbot_pattern = "chatbot = gr.Chatbot("
if chatbot_pattern in content:
    # Replace the chatbot definition with a simpler version
    content = content.replace(
        "chatbot = gr.Chatbot(",
        "chatbot = gr.Chatbot(value=[], elem_id=\"chatbot\", height=500, bubble_full_width=False, show_copy_button=True,"
    )
    
    # Write the modified content back to the file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Successfully simplified the chatbot interface")
    print(f"Fixed file written to: {OUTPUT_FILE}")
    print("\nTo use the fixed chatbot, run:")
    print(f"python {OUTPUT_FILE}")
else:
    print("Could not find chatbot definition in the file")
