import psycopg2
import pandas as pd

DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
}

def analyze_po_num_types():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    # Analyze po_followup_rev19
    cur.execute("SELECT po_num FROM public.po_followup_rev19 LIMIT 1000")
    rev19_po_nums = cur.fetchall()
    # Analyze po_followup_merged
    cur.execute('SELECT "PO_NUM" FROM po_data.po_followup_merged LIMIT 1000')
    merged_po_nums = cur.fetchall()
    conn.close()
    # Use pandas for type and value analysis
    rev19_df = pd.DataFrame(rev19_po_nums, columns=['po_num'])
    merged_df = pd.DataFrame(merged_po_nums, columns=['PO_NUM'])
    print("public.po_followup_rev19 sample types:")
    print(rev19_df.dtypes)
    print(rev19_df.head(10))
    print("po_data.po_followup_merged sample types:")
    print(merged_df.dtypes)
    print(merged_df.head(10))
    print("Sample intersection:")
    print(set(rev19_df['po_num']).intersection(set(merged_df['PO_NUM'])))

if __name__ == "__main__":
    analyze_po_num_types()
