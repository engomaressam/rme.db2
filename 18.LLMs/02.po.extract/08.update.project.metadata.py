import os
import mysql.connector
from mysql.connector import Error
import re

# Database configuration
DB_CONFIG = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

# Base path for projects
BASE_PATH = r'\\fileserver2\Head Office Server\Procurement (PR)\02 Projects Document\02-01 Finish'

# Document type keywords
CONTRACT_KEYWORDS = ['contract', 'contracts']
COMPARISON_KEYWORDS = ['comparison', 'comparisons', 'comparision']
PO_KEYWORDS = ['lpo', 'po', 'purchase', 'purchasing', 'order', 'tpd']

def connect_to_database():
    """Establish connection to MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        print("Connected to MySQL database")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def add_metadata_columns():
    """Add project_name and document_type columns to po.pdfs table if they don't exist"""
    connection = connect_to_database()
    if not connection:
        return False
    
    cursor = connection.cursor()
    
    try:
        # Check if columns exist
        cursor.execute("SHOW COLUMNS FROM `po.pdfs` LIKE 'project_name'")
        project_name_exists = cursor.fetchone() is not None
        
        cursor.execute("SHOW COLUMNS FROM `po.pdfs` LIKE 'document_type'")
        document_type_exists = cursor.fetchone() is not None
        
        # Add columns if they don't exist
        if not project_name_exists:
            cursor.execute("ALTER TABLE `po.pdfs` ADD COLUMN `project_name` VARCHAR(255)")
            print("Added project_name column to po.pdfs table")
        
        if not document_type_exists:
            cursor.execute("ALTER TABLE `po.pdfs` ADD COLUMN `document_type` VARCHAR(50)")
            print("Added document_type column to po.pdfs table")
        
        connection.commit()
        return True
    except Error as e:
        print(f"Error adding metadata columns: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def determine_document_type(folder_name):
    """Determine document type based on folder name"""
    folder_lower = folder_name.lower()
    
    # Check for contract
    for keyword in CONTRACT_KEYWORDS:
        if keyword in folder_lower:
            return 'Contract'
    
    # Check for comparison
    for keyword in COMPARISON_KEYWORDS:
        if keyword in folder_lower:
            return 'Comparison'
    
    # Check for PO
    for keyword in PO_KEYWORDS:
        if keyword in folder_lower:
            return 'Purchase Order'
    
    # Default if no match
    return 'Unknown'

def scan_project_structure():
    """Scan the project structure and return a dictionary of project metadata"""
    project_metadata = {}
    
    try:
        # Get all project folders
        project_folders = [f for f in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, f))]
        print(f"Found {len(project_folders)} project folders")
        
        for project_folder in project_folders:
            project_path = os.path.join(BASE_PATH, project_folder)
            
            # Get document type folders within the project
            try:
                document_folders = [f for f in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, f))]
                
                for doc_folder in document_folders:
                    doc_type = determine_document_type(doc_folder)
                    doc_path = os.path.join(project_path, doc_folder)
                    
                    # Walk through all files in this document folder
                    for root, dirs, files in os.walk(doc_path):
                        for file in files:
                            if file.lower().endswith('.pdf'):
                                full_path = os.path.join(root, file)
                                
                                # Store metadata
                                project_metadata[full_path] = {
                                    'project_name': project_folder,
                                    'document_type': doc_type
                                }
            except Exception as e:
                print(f"Error scanning project {project_folder}: {e}")
                continue
        
        print(f"Identified {len(project_metadata)} PDF files with metadata")
        return project_metadata
    except Exception as e:
        print(f"Error scanning project structure: {e}")
        return {}

def update_database_metadata(project_metadata):
    """Update the database with project and document type metadata"""
    connection = connect_to_database()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        # Get all PDFs from database
        cursor.execute("SELECT id, pdf_path FROM `po.pdfs` WHERE project_name IS NULL OR document_type IS NULL")
        pdf_records = cursor.fetchall()
        
        print(f"Found {len(pdf_records)} records in database that need metadata")
        
        # Track statistics
        updated_count = 0
        not_found_count = 0
        
        # Update each record
        for pdf_id, pdf_path in pdf_records:
            # Normalize path (handle different path formats)
            normalized_path = pdf_path.replace('/', '\\')
            
            # Try to find metadata for this path
            metadata = None
            
            # Direct match
            if normalized_path in project_metadata:
                metadata = project_metadata[normalized_path]
            else:
                # Try to find a partial match
                for path, meta in project_metadata.items():
                    # Extract the project name and file name parts
                    if os.path.basename(normalized_path) == os.path.basename(path):
                        # Check if the path contains the project name
                        if meta['project_name'] in normalized_path:
                            metadata = meta
                            break
            
            if metadata:
                # Update database
                cursor.execute(
                    "UPDATE `po.pdfs` SET project_name = %s, document_type = %s WHERE id = %s",
                    (metadata['project_name'], metadata['document_type'], pdf_id)
                )
                updated_count += 1
                
                # Print progress every 100 records
                if updated_count % 100 == 0:
                    print(f"Updated {updated_count} records so far...")
                    connection.commit()
            else:
                not_found_count += 1
        
        # Commit final changes
        connection.commit()
        
        print(f"Metadata update complete:")
        print(f"  - Updated: {updated_count} records")
        print(f"  - Not found: {not_found_count} records")
        
    except Error as e:
        print(f"Error updating database metadata: {e}")
    finally:
        cursor.close()
        connection.close()

def create_project_document_summary():
    """Create a summary table of projects and document types"""
    connection = connect_to_database()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        # Create summary table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS `po.project_summary` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            project_name VARCHAR(255) NOT NULL,
            document_type VARCHAR(50) NOT NULL,
            pdf_count INT NOT NULL,
            UNIQUE KEY (project_name, document_type)
        )
        """)
        
        # Clear existing data
        cursor.execute("TRUNCATE TABLE `po.project_summary`")
        
        # Insert summary data
        cursor.execute("""
        INSERT INTO `po.project_summary` (project_name, document_type, pdf_count)
        SELECT 
            project_name, 
            document_type, 
            COUNT(*) as pdf_count
        FROM `po.pdfs`
        WHERE project_name IS NOT NULL AND document_type IS NOT NULL
        GROUP BY project_name, document_type
        ORDER BY project_name, document_type
        """)
        
        connection.commit()
        
        # Get summary counts
        cursor.execute("SELECT COUNT(*) FROM `po.project_summary`")
        summary_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT project_name) FROM `po.project_summary`")
        project_count = cursor.fetchone()[0]
        
        print(f"Created project summary table with {summary_count} entries across {project_count} projects")
        
    except Error as e:
        print(f"Error creating project summary: {e}")
    finally:
        cursor.close()
        connection.close()

def main():
    print("Starting project metadata update process...")
    
    # Add metadata columns to database if they don't exist
    if not add_metadata_columns():
        print("Failed to add metadata columns. Exiting.")
        return
    
    # Scan project structure
    print("Scanning project structure...")
    project_metadata = scan_project_structure()
    
    if not project_metadata:
        print("No project metadata found. Exiting.")
        return
    
    # Update database with metadata
    print("Updating database with project metadata...")
    update_database_metadata(project_metadata)
    
    # Create project summary table
    print("Creating project summary table...")
    create_project_document_summary()
    
    print("Project metadata update process complete!")

if __name__ == "__main__":
    main()
