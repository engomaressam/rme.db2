import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

def fix_qty_unit_types():
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )
    cursor = conn.cursor()
    table = 'po_followup_rev19'
    schema = DB_CONFIG['schema']

    # Remove rows with non-numeric QUANTITY_RECEIVED or UNIT_PRICE (optional: comment out if you want to keep them as NULL)
    cursor.execute(f'''
        DELETE FROM {schema}."{table}"
        WHERE NOT ("QUANTITY_RECEIVED" ~ '^\\d+(\\.\\d+)?$')
           OR NOT ("UNIT_PRICE" ~ '^\\d+(\\.\\d+)?$');
    ''')
    print('Removed rows with non-numeric QUANTITY_RECEIVED or UNIT_PRICE.')

    # Alter column types
    cursor.execute(f'''
        ALTER TABLE {schema}."{table}"
        ALTER COLUMN "QUANTITY_RECEIVED" TYPE numeric USING "QUANTITY_RECEIVED"::numeric,
        ALTER COLUMN "UNIT_PRICE" TYPE numeric USING "UNIT_PRICE"::numeric;
    ''')
    print('Changed QUANTITY_RECEIVED and UNIT_PRICE to numeric.')

    conn.commit()
    cursor.close()
    conn.close()
    print('All done!')

if __name__ == "__main__":
    fix_qty_unit_types()
