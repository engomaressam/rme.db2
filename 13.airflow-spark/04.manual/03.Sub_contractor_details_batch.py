import os
import oracledb
import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Oracle ERP connection details
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

BATCH_SIZE = 10000
mysql_table = "Sub_contractor_details"
unique_key = "PO_DISTRIBUTION_ID"


def get_max_id_mysql():
    try:
        mysql_connection = mysql.connect(
            host=db_host, user=db_user, password=db_password, database=db_name
        )
        mysql_cursor = mysql_connection.cursor()
        mysql_cursor.execute(f"CREATE TABLE IF NOT EXISTS {mysql_table} ({unique_key} BIGINT PRIMARY KEY)")
        mysql_cursor.execute(f"SELECT MAX({unique_key}) FROM {mysql_table}")
        result = mysql_cursor.fetchone()[0]
        mysql_cursor.close()
        mysql_connection.close()
        return result if result is not None else 0
    except Error as e:
        print(f"❌ MySQL Error (get_max_id): {e}")
        return 0

def insert_to_mysql(df):
    try:
        mysql_connection = mysql.connect(
            host=db_host, user=db_user, password=db_password, database=db_name
        )
        mysql_cursor = mysql_connection.cursor()
        # Dynamically add new columns if needed
        existing_cols = set()
        mysql_cursor.execute(f"SHOW COLUMNS FROM {mysql_table}")
        for col in mysql_cursor.fetchall():
            existing_cols.add(col[0])
        for col in df.columns:
            if col not in existing_cols:
                mysql_cursor.execute(f"ALTER TABLE {mysql_table} ADD COLUMN `{col}` TEXT")
        # Insert, ignore duplicates
        placeholders = ", ".join(["%s"] * len(df.columns))
        insert_query = f"INSERT IGNORE INTO {mysql_table} ({', '.join(df.columns)}) VALUES ({placeholders})"
        data_tuples = [tuple(None if pd.isna(val) else val for val in row) for row in df.values]
        mysql_cursor.executemany(insert_query, data_tuples)
        mysql_connection.commit()
        print(f"✅ Inserted {len(df)} rows into MySQL.")
        mysql_cursor.close()
        mysql_connection.close()
    except Error as e:
        print(f"❌ MySQL Error (insert): {e}")

def fetch_batch_from_oracle(min_id):
    try:
        dsn = oracledb.makedsn(hostname, port, service_name=service_name)
        connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor_erp = connection_erp.cursor()
        query = f"""
            SELECT * FROM RME_DEV.Sub_contractor_details
            WHERE {unique_key} > :min_id
            ORDER BY {unique_key}
            FETCH NEXT {BATCH_SIZE} ROWS ONLY
        """
        cursor_erp.execute(query, {"min_id": min_id})
        columns = [col[0] for col in cursor_erp.description]
        data = cursor_erp.fetchall()
        df = pd.DataFrame(data, columns=columns)
        cursor_erp.close()
        connection_erp.close()
        return df
    except oracledb.Error as error:
        print(f"❌ Oracle Database error: {error}")
        return None

def main():
    min_id = get_max_id_mysql()
    print(f"Starting/resuming extraction from PO_DISTRIBUTION_ID > {min_id}")
    while True:
        df = fetch_batch_from_oracle(min_id)
        if df is None or df.empty:
            print("✅ Extraction complete. No more new rows.")
            break
        insert_to_mysql(df)
        min_id = df[unique_key].max()
        print(f"Next batch will start from PO_DISTRIBUTION_ID > {min_id}")

if __name__ == "__main__":
    main()
