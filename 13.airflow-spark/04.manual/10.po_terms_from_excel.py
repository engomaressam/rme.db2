import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

# Excel file path
excel_path = r"D:\OneDrive2\OneDrive - Rowad Modern Engineering\x004 Data Science\03.rme.db\01.tables\01.erp\04.po_followup\14.terms\all.po.terms.on.erp_202506101302.xlsx"

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

def insert_to_mysql(df, table_name):
    try:
        print("🔄 Connecting to MySQL database...")
        mysql_connection = mysql.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        mysql_cursor = mysql_connection.cursor()
        if mysql_connection.is_connected():
            print("✅ Successfully connected to MySQL!")

        print(f"🗑️ Dropping and recreating table {table_name}...")
        mysql_cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        # Create table with appropriate column types
        mysql_cursor.execute(f"""
            CREATE TABLE {table_name} (
                LONG_TEXT TEXT,
                PO_NUMBER VARCHAR(20)
            )
        """)
        mysql_connection.commit()

        print(f"📤 Inserting data into MySQL table {table_name} in batches...")
        placeholders = ", ".join(["%s"] * len(df.columns))
        insert_query = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({placeholders})"
        data_tuples = [tuple(None if pd.isna(val) else val for val in row) for row in df.values]
        batch_size = 1000
        for start in range(0, len(data_tuples), batch_size):
            end = start + batch_size
            mysql_cursor.executemany(insert_query, data_tuples[start:end])
            mysql_connection.commit()
        print(f"✅ Successfully inserted {len(df)} rows into MySQL.")
        mysql_cursor.close()
        mysql_connection.close()
    except Error as e:
        print(f"❌ MySQL Error: {e}")

def main():
    df = pd.read_excel(excel_path, dtype={"LONG_TEXT": str, "PO_NUMBER": str})
    # Remove any rows where both columns are NaN (e.g., empty rows at the end)
    df = df.dropna(how='all', subset=["LONG_TEXT", "PO_NUMBER"])
    # Ensure columns are named correctly
    df.columns = [col.strip() for col in df.columns]
    insert_to_mysql(df, "po_terms")

if __name__ == "__main__":
    main() 