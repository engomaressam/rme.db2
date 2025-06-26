import os
import re

def fix_process_question():
    """Fix the process_question function to improve project matching with proper variable scope"""
    file_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # Find the process_question function
    process_question_pattern = r'def process_question\(question: str, use_history: bool = True\).*?# Generate SQL query'
    process_question_match = re.search(process_question_pattern, code, re.DOTALL)
    
    if not process_question_match:
        print("Could not find process_question function in the code")
        return False
    
    # Get the current function code
    current_function = process_question_match.group(0)
    
    # Find the specific section we need to modify
    section_to_replace = """        # Process project if LLM detected one
        if "project" in llm_entities and llm_entities["project"] is not None:
            project_candidate = llm_entities["project"]
            
            # Exact match
            if project_candidate in UNIQUE_PROJECTS:
                detected_entities["project"] = project_candidate
                detected_entities["project_confidence"] = 1.0
                
                # Get alternative matches for reference
                other_matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                alternatives = [m for m in other_matches if m != project_candidate][:2]
                detected_entities["project_alternatives"] = alternatives
            
            # Fuzzy match
            else:
                matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                if matches:
                    detected_entities["project"] = matches[0]
                    detected_entities["project_confidence"] = difflib.SequenceMatcher(None, project_candidate, matches[0]).ratio()
                    detected_entities["project_alternatives"] = matches[1:3] if len(matches) > 1 else []"""
    
    # Create the improved section with special handling for Ring Road
    improved_section = """        # Process project if LLM detected one
        if "project" in llm_entities and llm_entities["project"] is not None:
            project_candidate = llm_entities["project"]
            
            # Exact match
            if project_candidate in UNIQUE_PROJECTS:
                detected_entities["project"] = project_candidate
                detected_entities["project_confidence"] = 1.0
                
                # Get alternative matches for reference
                other_matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                alternatives = [m for m in other_matches if m != project_candidate][:2]
                detected_entities["project_alternatives"] = alternatives
            
            # Special handling for Ring Road projects
            elif "ring road" in project_candidate.lower():
                ring_road_projects = [p for p in UNIQUE_PROJECTS if "ring road" in p.lower()]
                if ring_road_projects:
                    detected_entities["project"] = ring_road_projects[0]
                    detected_entities["project_confidence"] = 0.8
                    detected_entities["project_alternatives"] = ring_road_projects[1:3] if len(ring_road_projects) > 1 else []
                else:
                    # Fallback to regular fuzzy matching if no Ring Road projects found
                    matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                    if matches:
                        detected_entities["project"] = matches[0]
                        detected_entities["project_confidence"] = difflib.SequenceMatcher(None, project_candidate, matches[0]).ratio()
                        detected_entities["project_alternatives"] = matches[1:3] if len(matches) > 1 else []
            
            # Regular fuzzy matching for other projects
            else:
                matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)
                if matches:
                    detected_entities["project"] = matches[0]
                    detected_entities["project_confidence"] = difflib.SequenceMatcher(None, project_candidate, matches[0]).ratio()
                    detected_entities["project_alternatives"] = matches[1:3] if len(matches) > 1 else []"""
    
    # Replace the section in the function
    modified_function = current_function.replace(section_to_replace, improved_section)
    
    # Replace the function in the code
    modified_code = code.replace(current_function, modified_function)
    
    # Save the modified code to a new file
    new_file_path = file_path.replace(".py", ".final.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Updated code saved to {new_file_path}")
    return new_file_path

def print_debug_info():
    """Print debug info about the Ring Road projects"""
    file_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # Add debug code at the end of the file
    debug_code = """
# Add this at the end of the file to debug Ring Road detection
if __name__ == "__main__":
    # Print debug info about Ring Road projects before starting the server
    ring_road_projects = [p for p in UNIQUE_PROJECTS if "ring road" in p.lower()]
    print("\\n=== DEBUG: RING ROAD PROJECTS ===")
    print(f"Found {len(ring_road_projects)} Ring Road projects:")
    for p in ring_road_projects:
        print(f"  - {p}")
    print("=" * 40)
    
    # Continue with normal startup
    initialize_unique_lists()
    interface = create_interface()
    app = gr.mount_gradio_app(app, interface, path="/")
    webbrowser.open("http://localhost:7869", new=2, autoraise=True)
    uvicorn.run(app, host="0.0.0.0", port=7869)
"""
    
    # Replace the original main block
    main_pattern = r'if __name__ == "__main__":\s*initialize_unique_lists\(\)\s*interface = create_interface\(\)\s*app = gr\.mount_gradio_app\(app, interface, path="/"\)\s*.*\s*webbrowser\.open\("http://localhost:7869".*\)\s*uvicorn\.run\(app, host="0\.0\.0\.0", port=7869\)'
    modified_code = re.sub(main_pattern, debug_code, code)
    
    # Save the modified code to a new file
    new_file_path = file_path.replace(".py", ".debug.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Debug version saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Creating final fix for Ring Road project detection...")
    fixed_file = fix_process_question()
    
    print("\nCreating debug version to verify Ring Road projects...")
    debug_file = print_debug_info()
    
    print("\nFix completed! You have two new files:")
    print(f"1. {fixed_file} - The fixed version with proper Ring Road detection")
    print(f"2. {debug_file} - A debug version that prints Ring Road projects at startup")
    print("\nTo use the fixed version:")
    print(f"python \"{fixed_file}\"")
