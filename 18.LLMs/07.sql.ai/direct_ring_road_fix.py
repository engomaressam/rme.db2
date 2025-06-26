import re

def fix_process_question():
    """Create a direct fix for the Ring Road project detection issue"""
    file_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # Add a special case for Ring Road detection in extract_entity_from_question function
    extract_pattern = r'def extract_entity_from_question\(question: str, entity_type: str, entity_list: list\) -> Tuple\[Optional\[str\], float\]:'
    extract_function_start = re.search(extract_pattern, code)
    
    if not extract_function_start:
        print("Could not find extract_entity_from_question function")
        return False
    
    # Find the position right after the function signature
    pos = extract_function_start.end()
    
    # Add our special case code right after the function signature
    special_case_code = """
    # Special case for Ring Road projects
    if entity_type == "project" and "ring road" in question.lower():
        ring_road_projects = [entity for entity in entity_list if "ring road" in entity.lower()]
        if ring_road_projects:
            return ring_road_projects[0], 0.8
    """
    
    # Insert the special case code
    modified_code = code[:pos] + special_case_code + code[pos:]
    
    # Save the modified code to a new file
    new_file_path = file_path.replace(".py", ".ringroad.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Updated code saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Creating direct fix for Ring Road project detection...")
    fixed_file = fix_process_question()
    
    if fixed_file:
        print(f"\nFix completed! The updated code is in: {fixed_file}")
        print("\nTo use the fixed version:")
        print(f"python \"{fixed_file}\"")
        print("\nThis fix adds a special case that will detect 'ring road' in questions")
        print("and match it to any project containing 'ring road' in its name.")
    else:
        print("Failed to create the fix. Please check the file path.")
