import psycopg2

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres'
}

def list_all_tables():
    """List all tables in the database"""
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        cursor = conn.cursor()
        
        # Get all schemas
        cursor.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
            ORDER BY schema_name
        """)
        
        schemas = cursor.fetchall()
        
        for schema in schemas:
            schema_name = schema[0]
            print(f"\nSchema: {schema_name}")
            
            # Get all tables in this schema
            cursor.execute(f"""
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = '{schema_name}'
                ORDER BY table_name
            """)
            
            tables = cursor.fetchall()
            
            if tables:
                for table in tables:
                    table_name, table_type = table
                    # Count rows in the table
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")
                        row_count = cursor.fetchone()[0]
                        print(f"  {table_name} ({table_type}): {row_count} rows")
                    except:
                        print(f"  {table_name} ({table_type}): Error counting rows")
            else:
                print("  No tables found")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Listing all tables in the database...")
    list_all_tables()
