import sys
import os
import importlib.util
import json
import difflib
import re
from typing import Dict, List, Tuple, Optional, Any

def create_final_fix():
    """Create a comprehensive fix for project detection in rev.20"""
    rev16_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py"
    rev20_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the code from both files
    with open(rev16_path, 'r', encoding='utf-8') as file:
        rev16_code = file.read()
    
    with open(rev20_path, 'r', encoding='utf-8') as file:
        rev20_code = file.read()
    
    # Create a modified version of rev.20
    modified_code = rev20_code
    
    # 1. Replace the process_question function to handle project detection better
    process_pattern = r'def process_question\(question: str, use_history: bool = True\).*?# Generate SQL query'
    process_match = re.search(process_pattern, rev16_code, re.DOTALL)
    
    if process_match:
        rev16_process = process_match.group(0)
        # Update the rev.16 function to work with rev.20's table name
        rev16_process = rev16_process.replace("po_followup_merged", "po_followup_rev19")
        
        # Replace the function in rev.20
        modified_code = re.sub(process_pattern, rev16_process, modified_code, flags=re.DOTALL)
    
    # 2. Add a special case for project detection in extract_entity_from_question
    extract_pattern = r'def extract_entity_from_question\(question: str, entity_type: str, entity_list: list\) -> Tuple\[Optional\[str\], float\]:'
    extract_match = re.search(extract_pattern, modified_code)
    
    if extract_match:
        # Find the position right after the function signature
        pos = extract_match.end()
        
        # Add our special case code right after the function signature
        special_case_code = """
    # Special case for project detection
    if entity_type == "project":
        question_lower = question.lower()
        
        # Direct matching for project names
        for entity in entity_list:
            entity_lower = entity.lower()
            
            # Check for exact matches
            if entity_lower in question_lower:
                return entity, 1.0
            
            # Check for partial matches with common project words
            if "ring road" in question_lower and "ring road" in entity_lower:
                return entity, 0.9
            
            # Check for other common project patterns
            common_prefixes = ["project", "bridge", "road", "expansion", "extension"]
            for prefix in common_prefixes:
                if prefix in question_lower and prefix in entity_lower:
                    return entity, 0.8
    """
        
        # Insert the special case code
        modified_code = modified_code[:pos] + special_case_code + modified_code[pos:]
    
    # 3. Change the port number to avoid conflicts
    modified_code = modified_code.replace("port=7869", "port=7870")
    modified_code = modified_code.replace("http://localhost:7869", "http://localhost:7870")
    
    # 4. Lower the difflib cutoff threshold
    difflib_pattern = r'difflib\.get_close_matches\([^,]+,[^,]+, n=\d+, cutoff=(0\.\d+)\)'
    modified_code = re.sub(difflib_pattern, lambda m: m.group(0).replace(m.group(1), '0.4'), modified_code)
    
    # Save the modified code to a new file
    new_file_path = rev20_path.replace(".py", ".finalfix.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Final fix saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Creating comprehensive final fix for project detection...")
    fixed_file = create_final_fix()
    
    if fixed_file:
        print(f"\nFix completed! The updated code is in: {fixed_file}")
        print("\nTo use the fixed version:")
        print(f"python \"{fixed_file}\"")
        print("\nThis fix includes multiple improvements:")
        print("1. Uses rev.16's process_question function with rev.20's table name")
        print("2. Adds special case handling for project detection")
        print("3. Changes the port to 7870 to avoid conflicts")
        print("4. Lowers the difflib cutoff threshold to 0.4 for better matching")
