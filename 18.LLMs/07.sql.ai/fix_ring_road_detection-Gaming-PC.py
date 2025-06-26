import psycopg2
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

def connect_to_database():
    try:
        connection = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def find_ring_road_projects():
    """Find all projects containing 'Ring Road' in their name"""
    connection = None
    try:
        connection = connect_to_database()
        if not connection:
            return []
        
        cursor = connection.cursor()
        cursor.execute("""
            SELECT DISTINCT "PROJECT_NAME" 
            FROM po_followup_rev19 
            WHERE UPPER("PROJECT_NAME") LIKE '%RING ROAD%' 
            ORDER BY "PROJECT_NAME"
        """)
        results = [row[0] for row in cursor.fetchall() if row[0]]
        return results
    except Exception as e:
        print(f"Error finding Ring Road projects: {e}")
        return []
    finally:
        if connection:
            connection.close()

def update_entity_detection_code():
    """Update the entity detection code to improve Ring Road project matching"""
    file_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # Find the extract_entity_from_question function
    extract_entity_pattern = r'def extract_entity_from_question\(question: str, entity_type: str, entity_list: list\).*?return None, 0\.0'
    extract_entity_match = re.search(extract_entity_pattern, code, re.DOTALL)
    
    if not extract_entity_match:
        print("Could not find extract_entity_from_question function in the code")
        return False
    
    # Get the current function code
    current_function = extract_entity_match.group(0)
    
    # Modify the function to improve project detection for Ring Road
    improved_function = current_function.replace(
        '# Direct matching - more reliable for exact phrases in the question',
        '''# Direct matching - more reliable for exact phrases in the question
    
    # Special case for Ring Road project which is often mentioned without its full name
    if entity_type == "project" and "ring road" in question_lower:
        for entity in entity_list:
            if "ring road" in entity.lower():
                # Found a Ring Road project
                return entity, 1.0'''
    )
    
    # Replace the function in the code
    modified_code = code.replace(current_function, improved_function)
    
    # Save the modified code to a new file
    new_file_path = file_path.replace(".py", ".fixed.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Updated code saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Finding Ring Road projects in the database...")
    ring_road_projects = find_ring_road_projects()
    
    if ring_road_projects:
        print(f"Found {len(ring_road_projects)} Ring Road projects:")
        for project in ring_road_projects:
            print(f"  - {project}")
        
        print("\nUpdating entity detection code...")
        updated_file = update_entity_detection_code()
        
        if updated_file:
            print(f"\nFix completed. The updated code is in {updated_file}")
            print("\nTo use the fixed version:")
            print("1. Run the updated file instead of the original")
            print("2. The chatbot will now correctly detect Ring Road projects")
    else:
        print("No Ring Road projects found in the database. Please check your database connection and table.")
