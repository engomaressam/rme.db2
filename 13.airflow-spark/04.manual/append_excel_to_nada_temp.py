import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"
target_table = "RME_Projects_Cost_Dist_Line_Report_for_nada_temp"

excel_path = r"D:\ERP Downloads\cost_2025.xlsx"


def get_mysql_columns():
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cursor.execute(f"SHOW COLUMNS FROM {target_table}")
    columns = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return columns

def append_excel_to_mysql():
    # Load Excel
    df = pd.read_excel(excel_path)
    print(f"Excel shape: {df.shape}")
    # Get MySQL columns and align DataFrame
    mysql_cols = get_mysql_columns()
    df = df[[col for col in mysql_cols if col in df.columns]]
    print(f"Columns to insert: {list(df.columns)}")
    # Insert
    conn = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    cursor = conn.cursor()
    cols = [f'`{c}`' for c in df.columns]
    insert_sql = f"INSERT INTO {target_table} ({','.join(cols)}) VALUES ({','.join(['%s']*len(cols))})"
    cursor.executemany(insert_sql, df.astype(str).where(pd.notnull(df), None).values.tolist())
    conn.commit()
    print(f"✅ Inserted {len(df)} rows from Excel to {target_table}.")
    cursor.close()
    conn.close()

if __name__ == "__main__":
    append_excel_to_mysql()
