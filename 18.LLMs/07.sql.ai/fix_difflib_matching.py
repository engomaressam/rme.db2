import os
import re

# Fix the difflib matching in the process_question function

def fix_process_question():
    """Fix the process_question function to improve project matching"""
    file_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # Find the process_question function
    process_question_pattern = r'def process_question\(question: str, use_history: bool = True\).*?# Generate SQ'
    process_question_match = re.search(process_question_pattern, code, re.DOTALL)
    
    if not process_question_match:
        print("Could not find process_question function in the code")
        return False
    
    # Get the current function code
    current_function = process_question_match.group(0)
    
    # Modify the function to improve project detection for Ring Road
    improved_function = current_function.replace(
        '            # Fuzzy match\n            else:\n                matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)',
        '''            # Fuzzy match
            else:
                # Special handling for Ring Road projects
                if "ring road" in project_candidate.lower():
                    ring_road_projects = [p for p in UNIQUE_PROJECTS if "ring road" in p.lower()]
                    if ring_road_projects:
                        detected_entities["project"] = ring_road_projects[0]
                        detected_entities["project_confidence"] = 0.8
                        detected_entities["project_alternatives"] = ring_road_projects[1:3] if len(ring_road_projects) > 1 else []
                        # Skip the regular matching below
                        ring_road_detected = True
                    else:
                        ring_road_detected = False
                else:
                    ring_road_detected = False
                
                # Regular fuzzy matching if not a ring road or no ring road projects found
                if not ring_road_detected:
                    matches = difflib.get_close_matches(project_candidate, UNIQUE_PROJECTS, n=3, cutoff=0.5)'''
    )
    
    # Replace the function in the code
    modified_code = code.replace(current_function, improved_function)
    
    # Save the modified code to a new file
    new_file_path = file_path.replace(".py", ".fixed2.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Updated code saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Fixing the process_question function to improve project matching...")
    updated_file = fix_process_question()
    
    if updated_file:
        print(f"\nFix completed. The updated code is in {updated_file}")
        print("\nTo use the fixed version:")
        print("1. Run the updated file instead of the original")
        print("2. The chatbot will now correctly detect Ring Road projects")
    else:
        print("Failed to update the code. Please check the file path and try again.")
