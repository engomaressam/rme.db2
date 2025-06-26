import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

def print_table_types(table, schema):
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = %s
        ORDER BY ordinal_position
    """, (table, schema))
    print(f"\nSchema for {schema}.{table}:")
    for row in cursor.fetchall():
        print(f"{row[0]:25s} {row[1]}")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    print_table_types('po_followup_merged', 'po_data')
    print_table_types('po_followup_rev19', 'public')
