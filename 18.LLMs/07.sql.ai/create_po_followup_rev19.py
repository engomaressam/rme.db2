import psycopg2
import os

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    # 'schema': 'public',  # Not needed for psycopg2 connection
}

def create_rev19_table():
    create_sql = '''
    DROP TABLE IF EXISTS public.po_followup_rev19;
    CREATE TABLE public.po_followup_rev19 AS
    SELECT
        r17.project_name,
        r17.vendor AS vendor_name,
        r17.po_num,
        r17.po_comments AS comments,
        r17.approved_date,
        r17.uom,
        r17.description AS item_description,
        r17.unit_price,
        r17.qty_delivered AS quantity_received,
        r17.amount AS line_amount,
        m."TERMS" as terms
    FROM public.po_followup_rev17 r17
    LEFT JOIN po_data.po_followup_merged m
        ON r17.po_num = m."PO_NUM";
    '''
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(create_sql)
        print("Table po_followup_rev19 created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_rev19_table()
