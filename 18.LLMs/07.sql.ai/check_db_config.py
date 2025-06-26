import psycopg2
import sys

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres'
}

def check_table_details(schema, table):
    """Check the details of a table including structure, indexes, and constraints"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{schema}' 
                AND table_name = '{table}'
            )
        """)
        exists = cursor.fetchone()[0]
        
        if not exists:
            print(f"Table {schema}.{table} does not exist!")
            return
            
        print(f"Table {schema}.{table} exists.")
        
        # Get column information
        cursor.execute(f"""
            SELECT column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_schema = '{schema}'
            AND table_name = '{table}'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        print(f"\nColumns in {schema}.{table}:")
        for col in columns:
            col_name, data_type, max_length, nullable = col
            length_str = f"({max_length})" if max_length else ""
            nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
            print(f"  {col_name}: {data_type}{length_str} {nullable_str}")
        
        # Get primary key
        cursor.execute(f"""
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = '{schema}.{table}'::regclass
            AND i.indisprimary
        """)
        
        pk_columns = cursor.fetchall()
        if pk_columns:
            print("\nPrimary Key:")
            for pk in pk_columns:
                print(f"  {pk[0]}")
        else:
            print("\nNo Primary Key defined.")
        
        # Get indexes
        cursor.execute(f"""
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = '{schema}'
            AND tablename = '{table}'
        """)
        
        indexes = cursor.fetchall()
        if indexes:
            print("\nIndexes:")
            for idx in indexes:
                print(f"  {idx[0]}: {idx[1]}")
        else:
            print("\nNo indexes defined.")
        
        # Get constraints
        cursor.execute(f"""
            SELECT conname, pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_namespace n ON n.oid = c.connamespace
            WHERE conrelid = '{schema}.{table}'::regclass
            AND n.nspname = '{schema}'
        """)
        
        constraints = cursor.fetchall()
        if constraints:
            print("\nConstraints:")
            for con in constraints:
                print(f"  {con[0]}: {con[1]}")
        else:
            print("\nNo constraints defined.")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        row_count = cursor.fetchone()[0]
        print(f"\nRow count: {row_count}")
        
        # Get a sample row
        cursor.execute(f"SELECT * FROM {schema}.{table} LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print("\nSample row:")
            for i, col in enumerate(columns):
                print(f"  {col[0]}: {sample[i]}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check the original table
    print("=== CHECKING ORIGINAL TABLE ===")
    check_table_details("po_data", "po_followup_merged")
    
    # Check our copied table
    print("\n=== CHECKING COPIED TABLE ===")
    check_table_details("po_data", "merged2")
