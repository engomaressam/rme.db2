import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
}

def fix_po_num_format():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    # Update po_num to string without decimals
    update_sql = '''
    UPDATE public.po_followup_rev19
    SET po_num = regexp_replace(po_num::text, '\\.0$', '')
    WHERE po_num::text LIKE '%.0';
    '''
    cur.execute(update_sql)
    conn.commit()
    print("po_num format in rev19 fixed (no decimals).")
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_po_num_format()
