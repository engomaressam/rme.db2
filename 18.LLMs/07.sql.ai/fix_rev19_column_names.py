import psycopg2

# Update these with your actual DB credentials if different
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

# Mapping of current column names (as in rev.19) to desired names (as in rev.16/code)
# Only columns that need renaming should be listed here
COLUMN_RENAMES = {
    # 'current_name': 'DESIRED_NAME',
    # Example: 'vendor_name': 'VENDOR_NAME',
    # Fill in after inspecting your table structure
}

# List of all columns that should be UPPERCASE (as in code)
DESIRED_COLUMNS = [
    'PO_NUM', 'COMMENTS', 'APPROVED_DATE', 'UOM', 'ITEM_DESCRIPTION', 'UNIT_PRICE',
    'QUANTITY_RECEIVED', 'LINE_AMOUNT', 'PROJECT_NAME', 'VENDOR_NAME', 'TERMS'
]

def get_current_columns(cursor, table, schema):
    cursor.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s AND table_schema = %s
        ORDER BY ordinal_position
    """, (table, schema))
    return [row[0] for row in cursor.fetchall()]

def alter_column_names():
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )
    cursor = conn.cursor()
    table = 'po_followup_rev19'
    schema = DB_CONFIG['schema']
    
    current_columns = get_current_columns(cursor, table, schema)
    print('Current columns:', current_columns)

    # Build the renaming map automatically for lowercase to UPPERCASE
    rename_map = {}
    for col in current_columns:
        if col.upper() in DESIRED_COLUMNS and col != col.upper():
            rename_map[col] = col.upper()
    print('Columns to rename:', rename_map)

    for old, new in rename_map.items():
        print(f'Renaming {old} -> {new}')
        cursor.execute(f'ALTER TABLE {schema}."{table}" RENAME COLUMN "{old}" TO "{new}";')
    conn.commit()
    print('All done!')
    cursor.close()
    conn.close()

if __name__ == "__main__":
    alter_column_names()
