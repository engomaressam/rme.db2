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

def fetch_from_oracle(query):
    try:
        print("🔄 Connecting to the Oracle ERP database...")
        dsn = oracledb.makedsn(hostname, port, service_name=service_name)
        connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor_erp = connection_erp.cursor()
        print("✅ Successfully connected to Oracle ERP!")

        print("🔄 Running query on Oracle ERP...")
        cursor_erp.execute(query)
        columns = [col[0] for col in cursor_erp.description]
        data = cursor_erp.fetchall()
        # Convert LOBs to strings
        def convert_lob(val):
            if hasattr(val, 'read'):
                return val.read()
            return val
        data = [tuple(convert_lob(v) for v in row) for row in data]
        df = pd.DataFrame(data, columns=columns)
        print(f"✅ Fetched {len(df)} rows from Oracle ERP.")
        cursor_erp.close()
        connection_erp.close()
        return df
    except oracledb.Error as error:
        print(f"❌ Oracle Database error: {error}")
        return None

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
    oracle_query = "SELECT LONG_TEXT, PO_NUMBER FROM RME_DEV.XXRME_PO_TERMS_IN_TEXT"
    mysql_table = "po_terms"
    df = fetch_from_oracle(oracle_query)
    if df is not None and not df.empty:
        insert_to_mysql(df, mysql_table)
    else:
        print("❌ No data fetched from Oracle. Nothing to insert.")

if __name__ == "__main__":
    main() 