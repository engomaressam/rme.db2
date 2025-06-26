import psycopg2

DB_CONFIGS = [
    # (table, schema)
    ('po_followup_merged', 'po_data'),
    ('po_followup_rev19', 'public'),
]

PG_CREDS = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
}

def get_table_types(table, schema):
    conn = psycopg2.connect(**PG_CREDS)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = %s
        ORDER BY ordinal_position
    """, (table, schema))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return dict(results)

def compare_tables():
    types_1 = get_table_types(*DB_CONFIGS[0])
    types_2 = get_table_types(*DB_CONFIGS[1])
    print(f"{'Column':25s} {'Merged Type':15s} {'Rev19 Type':15s}")
    print("-"*60)
    all_cols = set(types_1) | set(types_2)
    for col in sorted(all_cols):
        t1 = types_1.get(col, '-')
        t2 = types_2.get(col, '-')
        print(f"{col:25s} {t1:15s} {t2:15s}")

if __name__ == "__main__":
    compare_tables()
