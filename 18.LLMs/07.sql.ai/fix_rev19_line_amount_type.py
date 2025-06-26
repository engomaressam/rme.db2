import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

def fix_line_amount_type():
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )
    cursor = conn.cursor()
    table = 'po_followup_rev19'
    schema = DB_CONFIG['schema']

    # Remove rows with non-numeric LINE_AMOUNT (optional: comment out if you want to keep them as NULL)
    cursor.execute(f'''
        DELETE FROM {schema}."{table}"
        WHERE NOT ("LINE_AMOUNT" ~ '^\\d+(\\.\\d+)?$');
    ''')
    print('Removed rows with non-numeric LINE_AMOUNT.')

    # Alter column type
    cursor.execute(f'''
        ALTER TABLE {schema}."{table}"
        ALTER COLUMN "LINE_AMOUNT" TYPE numeric USING "LINE_AMOUNT"::numeric;
    ''')
    print('Changed LINE_AMOUNT to numeric.')

    conn.commit()
    cursor.close()
    conn.close()
    print('All done!')

if __name__ == "__main__":
    fix_line_amount_type()
