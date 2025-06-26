import psycopg2
from psycopg2 import Error
import sys

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres'
}

# Table names
SOURCE_SCHEMA = 'public'
SOURCE_TABLE = 'po_followup_rev19'  # The rev.19 table with new data

TARGET_SCHEMA = 'po_data'  # The schema used by rev.16
TARGET_TABLE = 'po_followup_rev16_compatible'  # New table with rev.16 structure but rev.19 data

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

def get_table_structure(conn, schema, table):
    """Get the structure of a table"""
    cursor = conn.cursor()
    try:
        # Get column information
        cursor.execute(f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = '{schema}' AND table_name = '{table}'
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        
        if not columns:
            print(f"Error: Table {schema}.{table} does not exist or has no columns.")
            return None
            
        print(f"Found {len(columns)} columns in {schema}.{table}")
        return columns
    except Error as e:
        print(f"Error getting table structure: {e}")
        return None
    finally:
        cursor.close()

def create_new_table(conn, target_schema, target_table, columns):
    """Create a new table with the given structure"""
    cursor = conn.cursor()
    try:
        # Check if target schema exists, create if not
        cursor.execute(f"""
            SELECT schema_name FROM information_schema.schemata 
            WHERE schema_name = '{target_schema}'
        """)
        if not cursor.fetchone():
            print(f"Schema {target_schema} does not exist. Creating it...")
            cursor.execute(f"CREATE SCHEMA {target_schema}")
            conn.commit()
            print(f"Schema {target_schema} created successfully.")
        
        # Check if table already exists
        cursor.execute(f"""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = '{target_schema}' AND table_name = '{target_table}'
        """)
        if cursor.fetchone():
            print(f"Table {target_schema}.{target_table} already exists. Dropping it...")
            cursor.execute(f"DROP TABLE {target_schema}.{target_table}")
            conn.commit()
        
        # Create column definitions
        column_defs = []
        for col_name, data_type, max_length in columns:
            if max_length:
                column_def = f'"{col_name}" {data_type}({max_length})'
            else:
                column_def = f'"{col_name}" {data_type}'
            column_defs.append(column_def)
        
        # Create the table
        create_table_sql = f"""
            CREATE TABLE {target_schema}.{target_table} (
                {', '.join(column_defs)}
            )
        """
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"Table {target_schema}.{target_table} created successfully.")
        return True
    except Error as e:
        print(f"Error creating table: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()

def map_columns(source_columns, target_columns):
    """Map columns from source to target based on name"""
    source_col_names = [col[0] for col in source_columns]
    target_col_names = [col[0] for col in target_columns]
    
    # Find common columns
    common_columns = []
    for i, col_name in enumerate(target_col_names):
        if col_name in source_col_names:
            common_columns.append((col_name, source_col_names.index(col_name)))
    
    print(f"Found {len(common_columns)} common columns between source and target tables.")
    return common_columns

def copy_data(conn, source_schema, source_table, target_schema, target_table, column_mapping):
    """Copy data from source to target table for mapped columns"""
    cursor = conn.cursor()
    try:
        # Prepare column lists for SQL
        target_cols = [f'"{mapping[0]}"' for mapping in column_mapping]
        source_cols = [f'"{mapping[0]}"' for mapping in column_mapping]  # Use same column names
        
        # Insert data
        insert_sql = f"""
            INSERT INTO {target_schema}.{target_table} ({', '.join(target_cols)})
            SELECT {', '.join(source_cols)}
            FROM {source_schema}.{source_table}
        """
        cursor.execute(insert_sql)
        conn.commit()
        
        # Count rows
        cursor.execute(f"SELECT COUNT(*) FROM {target_schema}.{target_table}")
        row_count = cursor.fetchone()[0]
        print(f"Successfully copied {row_count} rows to {target_schema}.{target_table}")
        return row_count
    except Error as e:
        print(f"Error copying data: {e}")
        conn.rollback()
        return 0
    finally:
        cursor.close()

def main():
    # Connect to the database
    conn = create_connection()
    if not conn:
        print("Failed to connect to the database. Exiting.")
        sys.exit(1)
    
    try:
        # Step 1: Get the structure of both tables
        print(f"Getting structure of source table {SOURCE_SCHEMA}.{SOURCE_TABLE}...")
        source_columns = get_table_structure(conn, SOURCE_SCHEMA, SOURCE_TABLE)
        
        # Get the structure of the table used by rev.16 (po_followup_merged in po_data schema)
        REV16_SCHEMA = 'po_data'
        REV16_TABLE = 'po_followup_merged'
        print(f"Getting structure of rev.16 table {REV16_SCHEMA}.{REV16_TABLE}...")
        target_columns = get_table_structure(conn, REV16_SCHEMA, REV16_TABLE)
        
        if not source_columns or not target_columns:
            print("Failed to get table structures. Exiting.")
            sys.exit(1)
        
        # Step 2: Create the new table with rev.16 structure
        print(f"Creating new table {TARGET_SCHEMA}.{TARGET_TABLE} with rev.16 structure...")
        if not create_new_table(conn, TARGET_SCHEMA, TARGET_TABLE, target_columns):
            print("Failed to create new table. Exiting.")
            sys.exit(1)
        
        # Step 3: Map columns between source and target
        print("Mapping columns between source and target tables...")
        column_mapping = map_columns(source_columns, target_columns)
        if not column_mapping:
            print("No common columns found. Cannot copy data.")
            sys.exit(1)
        
        # Step 4: Copy data from source to target for mapped columns
        print(f"Copying data from {SOURCE_SCHEMA}.{SOURCE_TABLE} to {TARGET_SCHEMA}.{TARGET_TABLE}...")
        rows_copied = copy_data(conn, SOURCE_SCHEMA, SOURCE_TABLE, TARGET_SCHEMA, TARGET_TABLE, column_mapping)
        
        if rows_copied > 0:
            print(f"Success! Created {TARGET_SCHEMA}.{TARGET_TABLE} with {rows_copied} rows.")
            print(f"Update the chatbot to use this table by setting:")
            print(f"DB_CONFIG['schema'] = '{TARGET_SCHEMA}'")
            print(f"NEW_TABLE = '{TARGET_TABLE}'")
        else:
            print("Failed to copy data.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
