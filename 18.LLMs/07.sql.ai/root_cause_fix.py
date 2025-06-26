import sys
import os
import importlib.util
import json
import difflib
import re
from typing import Dict, List, Tuple, Optional, Any

def load_module_from_file(file_path, module_name):
    """Load a Python module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def compare_initialization():
    """Compare the initialization code between rev.16 and rev.20"""
    # Paths to the two versions
    rev16_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py"
    rev20_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the code from both files
    with open(rev16_path, 'r', encoding='utf-8') as file:
        rev16_code = file.read()
    
    with open(rev20_path, 'r', encoding='utf-8') as file:
        rev20_code = file.read()
    
    # Extract the initialization functions
    rev16_init = re.search(r'def initialize_unique_lists\(\).*?(?=\n\S|$)', rev16_code, re.DOTALL)
    rev20_init = re.search(r'def initialize_unique_lists\(\).*?(?=\n\S|$)', rev20_code, re.DOTALL)
    
    if rev16_init and rev20_init:
        rev16_init_code = rev16_init.group(0)
        rev20_init_code = rev20_init.group(0)
        
        print("=== COMPARING INITIALIZATION FUNCTIONS ===")
        
        # Split into lines for comparison
        rev16_lines = rev16_init_code.split('\n')
        rev20_lines = rev20_init_code.split('\n')
        
        # Find differences
        differences = []
        for i, (line16, line20) in enumerate(zip(rev16_lines, rev20_lines)):
            if line16 != line20:
                differences.append((i+1, line16, line20))
        
        # Handle different lengths
        if len(rev16_lines) > len(rev20_lines):
            for i in range(len(rev20_lines), len(rev16_lines)):
                differences.append((i+1, rev16_lines[i], "MISSING IN REV.20"))
        elif len(rev20_lines) > len(rev16_lines):
            for i in range(len(rev16_lines), len(rev20_lines)):
                differences.append((i+1, "MISSING IN REV.16", rev20_lines[i]))
        
        if differences:
            print(f"Found {len(differences)} differences in initialization:")
            for line_num, line16, line20 in differences:
                print(f"Line {line_num}:")
                print(f"  Rev.16: {line16}")
                print(f"  Rev.20: {line20}")
                print()
        else:
            print("No differences found in initialization code")
    else:
        print("Could not extract initialization functions")

def create_root_cause_fix():
    """Create a fix that addresses the root cause of the entity detection differences"""
    rev16_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py"
    rev20_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the code from both files
    with open(rev16_path, 'r', encoding='utf-8') as file:
        rev16_code = file.read()
    
    with open(rev20_path, 'r', encoding='utf-8') as file:
        rev20_code = file.read()
    
    # Modify the difflib cutoff value in rev.20 to be more lenient
    modified_code = rev20_code
    
    # Find all instances of difflib.get_close_matches with a cutoff parameter
    difflib_pattern = r'difflib\.get_close_matches\([^,]+,[^,]+, n=\d+, cutoff=(0\.\d+)\)'
    
    # Replace the cutoff with a more lenient value (0.4 instead of 0.5)
    modified_code = re.sub(difflib_pattern, lambda m: m.group(0).replace(m.group(1), '0.4'), modified_code)
    
    # Modify the extract_entity_from_question function to improve matching
    extract_pattern = r'def extract_entity_from_question\(question: str, entity_type: str, entity_list: list\) -> Tuple\[Optional\[str\], float\]:'
    extract_function_start = re.search(extract_pattern, modified_code)
    
    if extract_function_start:
        # Find the position right after the function signature
        pos = extract_function_start.end()
        
        # Add our improved matching code right after the function signature
        improved_matching_code = """
    # Improved matching for better entity detection
    question_lower = question.lower()
    
    # Direct matching - more reliable for exact phrases in the question
    for entity in entity_list:
        if entity.lower() in question_lower:
            return entity, 1.0
    
    # Try matching with common project prefixes/suffixes removed
    common_terms = ["project", "bridge", "road", "expansion", "extension"]
    for entity in entity_list:
        entity_lower = entity.lower()
        # Create simplified versions of the entity name
        simplified_entities = [entity_lower]
        
        for term in common_terms:
            if entity_lower.startswith(term):
                simplified_entities.append(entity_lower[len(term):].strip())
            if entity_lower.endswith(term):
                simplified_entities.append(entity_lower[:-len(term)].strip())
            if f" {term} " in entity_lower:
                simplified_entities.append(entity_lower.replace(f" {term} ", " ").strip())
        
        # Check if any simplified version is in the question
        for simple_entity in simplified_entities:
            if simple_entity and simple_entity in question_lower:
                return entity, 0.9
    """
        
        # Find the original function body
        function_body_pattern = r'    question_lower = question\.lower\(\)\s+# Direct matching.*?return None, 0\.0'
        function_body_match = re.search(function_body_pattern, modified_code, re.DOTALL)
        
        if function_body_match:
            # Replace the original function body with our improved version
            modified_code = modified_code[:pos] + improved_matching_code + modified_code[function_body_match.end():]
        else:
            # If we can't find the exact function body, just add our code after the signature
            modified_code = modified_code[:pos] + improved_matching_code + modified_code[pos:]
    
    # Save the modified code to a new file
    new_file_path = rev20_path.replace(".py", ".rootfix.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Root cause fix saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Analyzing root cause of entity detection differences...\n")
    compare_initialization()
    
    print("\nCreating root cause fix...")
    fixed_file = create_root_cause_fix()
    
    if fixed_file:
        print(f"\nFix completed! The updated code is in: {fixed_file}")
        print("\nTo use the fixed version:")
        print(f"python \"{fixed_file}\"")
        print("\nThis fix addresses the root cause of the entity detection differences by:")
        print("1. Making the fuzzy matching more lenient (lowering the cutoff from 0.5 to 0.4)")
        print("2. Adding improved entity matching that handles common project naming variations")
        print("3. Preserving the original code structure while enhancing its capabilities")
