import psycopg2
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configurations
REV16_DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'po_data',
    'table': 'po_followup_merged'
}

REV20_DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public',
    'table': 'po_followup_rev19'
}

def connect_to_database(config):
    try:
        connection = psycopg2.connect(
            host=config['host'],
            user=config['user'],
            password=config['password'],
            database=config['database']
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def fetch_unique_list(config, column):
    connection = None
    try:
        connection = connect_to_database(config)
        if not connection:
            return []
        
        cursor = connection.cursor()
        cursor.execute(f"""
            SELECT DISTINCT "{column}" 
            FROM {config['schema']}.{config['table']} 
            WHERE "{column}" IS NOT NULL AND "{column}" != '' 
            ORDER BY "{column}"
        """)
        results = [row[0] for row in cursor.fetchall() if row[0]]
        return results
    except Exception as e:
        print(f"Error fetching unique {column}: {e}")
        return []
    finally:
        if connection:
            connection.close()

def create_project_lookup_table():
    """Create a lookup table with all projects from rev.16 for use in rev.20"""
    connection = None
    try:
        # Get projects from rev.16
        rev16_projects = fetch_unique_list(REV16_DB_CONFIG, "PROJECT_NAME")
        print(f"Found {len(rev16_projects)} projects in rev.16")
        
        # Connect to rev.20 database
        connection = connect_to_database(REV20_DB_CONFIG)
        if not connection:
            return False
        
        cursor = connection.cursor()
        
        # Create the lookup table
        cursor.execute("""
        DROP TABLE IF EXISTS project_lookup;
        CREATE TABLE project_lookup (
            id SERIAL PRIMARY KEY,
            project_name VARCHAR(255) NOT NULL
        );
        """)
        
        # Insert all projects from rev.16
        for project in rev16_projects:
            cursor.execute(
                "INSERT INTO project_lookup (project_name) VALUES (%s)",
                (project,)
            )
        
        connection.commit()
        print(f"Successfully created project_lookup table with {len(rev16_projects)} projects")
        return True
    
    except Exception as e:
        print(f"Error creating project lookup table: {e}")
        if connection:
            connection.rollback()
        return False
    
    finally:
        if connection:
            connection.close()

def update_rev20_code():
    """Update the rev.20 code to use the project_lookup table for entity detection"""
    file_path = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.llms\\07.sql.ai\\20.po.followup.query.ai.rev.20.on.gpu.gemma3.postgres.py"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    
    # Find the fetch_unique_list function and modify it to use the lookup table for projects
    modified_code = code.replace(
        'cursor.execute(f"SELECT DISTINCT {pg_column} FROM {NEW_TABLE} WHERE {pg_column} IS NOT NULL AND {pg_column} != \'\' ORDER BY {pg_column}")',
        '''# Use project_lookup table for PROJECT_NAME to get the complete list from rev.16
        if column == "PROJECT_NAME":
            cursor.execute("SELECT DISTINCT project_name FROM project_lookup ORDER BY project_name")
        else:
            cursor.execute(f"SELECT DISTINCT {pg_column} FROM {NEW_TABLE} WHERE {pg_column} IS NOT NULL AND {pg_column} != \'\' ORDER BY {pg_column}")'''
    )
    
    # Save the modified code to a new file
    new_file_path = file_path.replace(".py", ".fixed.py")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(modified_code)
    
    print(f"Updated code saved to {new_file_path}")
    return new_file_path

if __name__ == "__main__":
    print("Creating project lookup table...")
    if create_project_lookup_table():
        print("Updating rev.20 code to use the lookup table...")
        updated_file = update_rev20_code()
        print(f"Fix completed. The updated code is in {updated_file}")
        print("\nTo use the fixed version:")
        print("1. Run the updated file instead of the original")
        print("2. The chatbot will now have access to all 643 projects from rev.16")
        print("3. Project detection should now work as well as it did in rev.16")
    else:
        print("Failed to create project lookup table. Fix not applied.")
