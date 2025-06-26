import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text

# --- CONFIG ---
EXCEL_PATH = r"C:\Users\Omar Essam2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\00.repo\rme.db\18.llms\07.sql.ai\01.erp.table\RME_PO_Follow_up_180625.xlsx"
DB_PARAMS = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres',
    'schema': 'public',
}
# Table names
table_erp = 'po_followup_from_erp'
table_rev17 = 'po_followup_rev17'

# --- 1. Check Excel ---
def check_excel():
    df = pd.read_excel(EXCEL_PATH)
    print(f"Excel columns: {list(df.columns)}")
    mask = df['Project Name'].str.contains('ring road', case=False, na=False)
    filtered = df[mask]
    print(f"[EXCEL] Rows with 'ring road' in Project Name: {filtered.shape[0]}")
    print(filtered[['Project Name']].drop_duplicates())
    return filtered

# --- 2. Check PostgreSQL tables ---
def get_pg_engine():
    import urllib.parse
    user = urllib.parse.quote_plus(DB_PARAMS['user'])
    password = urllib.parse.quote_plus(DB_PARAMS['password'])
    url = f"postgresql+psycopg2://{user}:{password}@{DB_PARAMS['host']}/{DB_PARAMS['database']}"
    return create_engine(url)

def check_pg_table(table):
    engine = get_pg_engine()
    with engine.connect() as conn:
        sql = text(f"SELECT * FROM {table} WHERE project_name ILIKE '%ring road%'")
        df = pd.read_sql(sql, conn)
        print(f"[TABLE: {table}] Rows with 'ring road' in project_name: {df.shape[0]}")
        print(df[['project_name']].drop_duplicates())
        return df

if __name__ == "__main__":
    print("--- Checking Excel file ---")
    check_excel()
    print("\n--- Checking po_followup_from_erp table ---")
    check_pg_table(table_erp)
    print("\n--- Checking po_followup_rev17 table ---")
    check_pg_table(table_rev17)
