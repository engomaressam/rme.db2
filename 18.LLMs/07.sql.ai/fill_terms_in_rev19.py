import psycopg2

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
}

def fill_terms():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    # Update terms in rev19 from merged table, matching as text
    update_sql = '''
    UPDATE public.po_followup_rev19 AS r19
    SET terms = m."TERMS"
    FROM po_data.po_followup_merged AS m
    WHERE CAST(r19.po_num AS TEXT) = CAST(m."PO_NUM" AS TEXT)
    '''
    cur.execute(update_sql)
    conn.commit()
    print("Terms updated in rev19 from merged table.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    fill_terms()
