import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public'
}

def check_max_unit_price(project_name):
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database=DB_CONFIG['database']
    )
    cursor = conn.cursor()
    table = 'po_followup_rev19'
    schema = DB_CONFIG['schema']
    
    cursor.execute(f'''
        SELECT "UNIT_PRICE", "ITEM_DESCRIPTION", "VENDOR_NAME"
        FROM {schema}."{table}"
        WHERE "PROJECT_NAME" = %s
        ORDER BY "UNIT_PRICE" DESC
        LIMIT 10;
    ''', (project_name,))
    rows = cursor.fetchall()
    print(f"Top 10 unit prices for project: {project_name}")
    print(f"{'UNIT_PRICE':>12} | {'ITEM_DESCRIPTION':20s} | {'VENDOR_NAME'}")
    print('-'*60)
    # Print ASCII-only summary to console
    for price, item, vendor in rows:
        try:
            ascii_item = item.encode('ascii', errors='ignore').decode('ascii')
            ascii_vendor = vendor.encode('ascii', errors='ignore').decode('ascii')
            print(f"{price:12} | {ascii_item[:20]:20s} | {ascii_vendor}")
        except Exception:
            print(f"{price:12} | <non-ascii> | <non-ascii>")
    # Write full results to a UTF-8 file
    with open('max_unit_prices.txt', 'w', encoding='utf-8') as f:
        f.write(f"Top 10 unit prices for project: {project_name}\n")
        f.write(f"{'UNIT_PRICE':>12} | {'ITEM_DESCRIPTION':50s} | {'VENDOR_NAME'}\n")
        f.write('-'*90 + '\n')
        for price, item, vendor in rows:
            f.write(f"{price:12} | {item[:50]:50s} | {vendor}\n")
    cursor.close()
    conn.close()
    print('Full results written to max_unit_prices.txt')

if __name__ == "__main__":
    check_max_unit_price('EGAT Pelletizing Plant-0144')
