import os
import re

# File paths
SOURCE_FILE = "16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py"
TARGET_FILE = "24.po.followup.query.ai.rev.24.on.gpu.gemma3.postgres.py"

# Table names
OLD_TABLE = "po_followup_merged"
NEW_TABLE = "merged2"

# Schema names (in case they need to be replaced)
OLD_SCHEMA = "po_data"
NEW_SCHEMA = "po_data"  # Same in this case

def replace_table_references():
    """Create a new file with all table references replaced"""
    # First, copy the file
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as source:
            content = source.read()
        
        # Replace the table name variable
        content = re.sub(r'NEW_TABLE\s*=\s*["\']po_followup_merged["\']', f'NEW_TABLE = "{NEW_TABLE}"', content)
        
        # Replace any hardcoded table references in strings
        content = content.replace(f"'{OLD_TABLE}'", f"'{NEW_TABLE}'")
        content = content.replace(f'"{OLD_TABLE}"', f'"{NEW_TABLE}"')
        content = content.replace(f"table name is '{OLD_TABLE}'", f"table name is '{NEW_TABLE}'")
        content = content.replace(f"table {OLD_TABLE}", f"table {NEW_TABLE}")
        content = content.replace(f"FROM {OLD_TABLE}", f"FROM {NEW_TABLE}")
        content = content.replace(f"from {OLD_TABLE}", f"from {NEW_TABLE}")
        
        # Update the port to avoid conflicts
        content = content.replace("port=7869", "port=7871")
        content = content.replace('"http://localhost:7869"', '"http://localhost:7871"')
        
        # Update the title to reflect the new version
        content = content.replace("(rev.16,", "(rev.24,")
        
        # Write to the new file
        with open(TARGET_FILE, 'w', encoding='utf-8') as target:
            target.write(content)
            
        print(f"Successfully created {TARGET_FILE} with all table references updated.")
        return True
    except Exception as e:
        print(f"Error replacing table references: {e}")
        return False

if __name__ == "__main__":
    if replace_table_references():
        print("Table name replacement completed successfully.")
    else:
        print("Failed to replace table names.")
