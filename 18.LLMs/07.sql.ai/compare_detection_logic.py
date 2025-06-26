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

def extract_function_text(code, function_name):
    """Extract the full text of a function from code"""
    pattern = fr'def {function_name}\([^)]*\).*?(?=\n\S|$)'
    match = re.search(pattern, code, re.DOTALL)
    if match:
        return match.group(0)
    return None

def compare_functions():
    """Compare the entity detection functions between rev.16 and rev.20"""
    # Paths to the two versions
    rev16_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py"
    rev20_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the code from both files
    with open(rev16_path, 'r', encoding='utf-8') as file:
        rev16_code = file.read()
    
    with open(rev20_path, 'r', encoding='utf-8') as file:
        rev20_code = file.read()
    
    # Functions to compare
    functions = [
        "detect_entities_with_llm",
        "extract_entity_from_question",
        "process_question"
    ]
    
    # Compare each function
    print("=== DETAILED FUNCTION COMPARISON ===\n")
    
    for func_name in functions:
        print(f"Comparing function: {func_name}")
        
        rev16_func = extract_function_text(rev16_code, func_name)
        rev20_func = extract_function_text(rev20_code, func_name)
        
        if not rev16_func or not rev20_func:
            print(f"  Could not extract function {func_name} from one or both files")
            continue
        
        # Split into lines for comparison
        rev16_lines = rev16_func.split('\n')
        rev20_lines = rev20_func.split('\n')
        
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
            print(f"  Found {len(differences)} differences:")
            for line_num, line16, line20 in differences:
                print(f"  Line {line_num}:")
                print(f"    Rev.16: {line16}")
                print(f"    Rev.20: {line20}")
                print()
        else:
            print("  No differences found in function code")
        
        print("-" * 80)

def create_comprehensive_fix():
    """Create a comprehensive fix based on the differences"""
    rev16_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py"
    rev20_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the code from both files
    with open(rev16_path, 'r', encoding='utf-8') as file:
        rev16_code = file.read()
    
    with open(rev20_path, 'r', encoding='utf-8') as file:
        rev20_code = file.read()
    
    # Extract the functions from rev.16
    rev16_extract_entity = extract_function_text(rev16_code, "extract_entity_from_question")
    rev16_process_question = extract_function_text(rev16_code, "process_question")
    
    if not rev16_extract_entity or not rev16_process_question:
        print("Could not extract required functions from rev.16")
        return False
    
    # Create a new version of rev.20 with the rev.16 functions
    new_code = rev20_code
    
    # Replace extract_entity_from_question
    extract_pattern = r'def extract_entity_from_question\([^)]*\).*?(?=\n\S|$)'
    new_code = re.sub(extract_pattern, rev16_extract_entity, new_code, flags=re.DOTALL)
    
    # Replace process_question
    process_pattern = r'def process_question\([^)]*\).*?(?=\n\S|$)'
    new_code = re.sub(process_pattern, rev16_process_question, new_code, flags=re.DOTALL)
    
    # Save the modified code to a new file
    new_file_path = rev20_path.replace(".py", ".comprehensive.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(new_code)
    
    print(f"Comprehensive fix saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Comparing entity detection functions between rev.16 and rev.20...\n")
    compare_functions()
    
    print("\nCreating comprehensive fix that ports rev.16 entity detection to rev.20...")
    fixed_file = create_comprehensive_fix()
    
    if fixed_file:
        print(f"\nFix completed! The updated code is in: {fixed_file}")
        print("\nTo use the fixed version:")
        print(f"python \"{fixed_file}\"")
        print("\nThis fix replaces the entity detection functions in rev.20 with those from rev.16,")
        print("ensuring that all projects are detected correctly, not just Ring Road.")
