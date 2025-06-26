import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
}

def list_columns():
    query = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'po_data'
          AND table_name = 'po_followup_merged';
    """
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(query)
        columns = [r[0] for r in cur.fetchall()]
    conn.close()
    print("Columns in po_data.po_followup_merged:")
    for col in columns:
        print(col)

if __name__ == "__main__":
    list_columns()
