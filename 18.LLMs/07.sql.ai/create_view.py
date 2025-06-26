import psycopg2
from psycopg2 import Error

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres'
}

def create_connection():
    """Create a connection to the PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        return conn
    except Error as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def create_view():
    """Create a view that points to the original table"""
    conn = create_connection()
    if not conn:
        print("Failed to connect to the database. Exiting.")
        return False
    
    cursor = conn.cursor()
    try:
        # Check if view already exists
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'po_data' AND table_name = 'po_followup_merged'
        """)
        if not cursor.fetchone():
            print("Error: Original table po_data.po_followup_merged does not exist.")
            return False
            
        # Drop the view if it exists
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'po_data' AND table_name = 'po_followup_merged_view'
        """)
        if cursor.fetchone():
            print("View po_data.po_followup_merged_view already exists. Dropping it...")
            cursor.execute("DROP VIEW po_data.po_followup_merged_view")
            conn.commit()
        
        # Create the view
        print("Creating view po_data.po_followup_merged_view...")
        cursor.execute("""
            CREATE VIEW po_data.po_followup_merged_view AS
            SELECT * FROM po_data.po_followup_merged
        """)
        conn.commit()
        
        print("View created successfully.")
        return True
    except Error as e:
        print(f"Error creating view: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    if create_view():
        print("View creation completed successfully.")
        print("Update the chatbot to use this view by setting:")
        print("NEW_TABLE = \"po_followup_merged_view\"")
    else:
        print("Failed to create view.")
