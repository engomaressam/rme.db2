import psycopg2
from psycopg2 import Error

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres'
}

# Table names
SOURCE_SCHEMA = 'po_data'
SOURCE_TABLE = 'po_followup_merged'  # The original table used by rev.16

TARGET_SCHEMA = 'po_data'  # Same schema
TARGET_TABLE = 'merged2'  # New table name

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

def copy_table_structure_and_data():
    """Create an exact copy of the source table with all data"""
    conn = create_connection()
    if not conn:
        print("Failed to connect to the database. Exiting.")
        return False
    
    cursor = conn.cursor()
    try:
        # Check if target table already exists
        cursor.execute(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = '{TARGET_SCHEMA}' AND table_name = '{TARGET_TABLE}'
        """)
        if cursor.fetchone():
            print(f"Table {TARGET_SCHEMA}.{TARGET_TABLE} already exists. Dropping it...")
            cursor.execute(f"DROP TABLE {TARGET_SCHEMA}.{TARGET_TABLE}")
            conn.commit()
        
        # Create the new table as an exact copy of the source table
        print(f"Creating new table {TARGET_SCHEMA}.{TARGET_TABLE} as a copy of {SOURCE_SCHEMA}.{SOURCE_TABLE}...")
        cursor.execute(f"CREATE TABLE {TARGET_SCHEMA}.{TARGET_TABLE} AS TABLE {SOURCE_SCHEMA}.{SOURCE_TABLE}")
        conn.commit()
        
        # Count rows in the new table
        cursor.execute(f"SELECT COUNT(*) FROM {TARGET_SCHEMA}.{TARGET_TABLE}")
        row_count = cursor.fetchone()[0]
        print(f"Successfully created {TARGET_SCHEMA}.{TARGET_TABLE} with {row_count} rows.")
        
        return True
    except Error as e:
        print(f"Error copying table: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    if copy_table_structure_and_data():
        print("Table copy completed successfully.")
    else:
        print("Failed to copy table.")
