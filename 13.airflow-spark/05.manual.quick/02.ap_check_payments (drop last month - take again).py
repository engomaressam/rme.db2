import os
import oracledb
import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from datetime import datetime, timedelta

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
mysql_table = "RME_ap_check_payments_Report"

# Date range
START_YEAR = 2016
START_MONTH = 1


def get_latest_month_in_mysql():
    try:
        mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        mysql_cursor = mysql_connection.cursor()
        mysql_cursor.execute(f"SELECT MAX(CHECK_DATE) FROM {mysql_table}")
        result = mysql_cursor.fetchone()[0]
        print(f"[DEBUG] MAX(CHECK_DATE) result: {result} (type: {type(result)})")
        mysql_cursor.close()
        mysql_connection.close()
        if result is not None:
            # Parse string to datetime if needed
            import pandas as pd
            if isinstance(result, str):
                result_dt = pd.to_datetime(result)
            else:
                result_dt = result
            start_date = datetime(result_dt.year, result_dt.month, 1)
            print(f"[DEBUG] Returning start_date: {start_date}")
            return start_date
        else:
            print(f"[DEBUG] No data found, returning default start: {datetime(START_YEAR, START_MONTH, 1)}")
            return datetime(START_YEAR, START_MONTH, 1)
    except Exception as e:
        print(f"❌ MySQL error (get_latest_month): {e}")
        return datetime(START_YEAR, START_MONTH, 1)

def ensure_table_exists(df):
    if df is None or df.empty:
        return
    try:
        mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        mysql_cursor = mysql_connection.cursor()
        col_defs = ", ".join([f"`{col}` TEXT" for col in df.columns])
        mysql_cursor.execute(f"CREATE TABLE IF NOT EXISTS {mysql_table} ({col_defs})")
        mysql_connection.commit()
        mysql_cursor.close()
        mysql_connection.close()
    except Exception as e:
        print(f"❌ MySQL error (ensure_table_exists): {e}")

def delete_month_from_mysql(year, month, df):
    if df is None or df.empty:
        return
    ensure_table_exists(df)
    try:
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year+1, 1, 1)
        else:
            end_date = datetime(year, month+1, 1)
        mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        mysql_cursor = mysql_connection.cursor()
        print(f"🗑️ Deleting rows for {start_date.strftime('%b-%Y')} from MySQL...")
        mysql_cursor.execute(f"DELETE FROM {mysql_table} WHERE CHECK_DATE >= %s AND CHECK_DATE < %s", (start_date, end_date))
        mysql_connection.commit()
        mysql_cursor.close()
        mysql_connection.close()
    except Exception as e:
        print(f"❌ MySQL error (delete_month): {e}")

def insert_to_mysql(df):
    if df is None or df.empty:
        return
    ensure_table_exists(df)
    try:
        mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        mysql_cursor = mysql_connection.cursor()
        # Insert
        columns = [f"`{col}`" for col in df.columns]
        placeholders = ", ".join(["%s"] * len(df.columns))
        insert_query = f"INSERT INTO {mysql_table} ({', '.join(columns)}) VALUES ({placeholders})"
        data_tuples = [tuple(None if pd.isna(val) else val for val in row) for row in df.values]
        mysql_cursor.executemany(insert_query, data_tuples)
        mysql_connection.commit()
        print(f"✅ Inserted {len(df)} rows into MySQL.")
        mysql_cursor.close()
        mysql_connection.close()
    except Error as e:
        print(f"❌ MySQL Error (insert): {e}")

def fetch_month_from_oracle(year, month):
    try:
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year+1, 1, 1)
        else:
            end_date = datetime(year, month+1, 1)
        dsn = oracledb.makedsn(hostname, port, service_name=service_name)
        connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor_erp = connection_erp.cursor()
        print(f"🔄 Fetching data for {start_date.strftime('%b-%Y')} from Oracle...")
        query = f"""
            SELECT * FROM RME_DEV.RME_AP_CHECK_PAYMENTS
            WHERE CHECK_DATE >= :start_date AND CHECK_DATE < :end_date
        """
        cursor_erp.execute(query, {"start_date": start_date, "end_date": end_date})
        columns = [col[0] for col in cursor_erp.description]
        data = cursor_erp.fetchall()
        df = pd.DataFrame(data, columns=columns)
        print(f"✅ Oracle: {len(df)} rows fetched for {start_date.strftime('%b-%Y')}")
        cursor_erp.close()
        connection_erp.close()
        return df
    except oracledb.Error as error:
        print(f"❌ Oracle Database error: {error}")
        return None

def main():
    # Find latest month in MySQL, or start from Jan 2016
    latest_date = get_latest_month_in_mysql()
    print(f"[INFO] Detected start date for extraction: {latest_date} (year={latest_date.year}, month={latest_date.month})")
    year = latest_date.year
    month = latest_date.month
    now = datetime.now()
    while (year < now.year) or (year == now.year and month <= now.month):
        print(f"\n=== Processing {datetime(year, month, 1).strftime('%b-%Y')} ===")
        df = fetch_month_from_oracle(year, month)
        if df is not None and not df.empty:
            print(f"⏳ DataFrame shape: {df.shape}. Preview:\n{df.head(3)}")
            delete_month_from_mysql(year, month, df)
            insert_to_mysql(df)
        else:
            print(f"❌ No data fetched from Oracle for {datetime(year, month, 1).strftime('%b-%Y')}. Skipping insert.")
        # Move to next month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

if __name__ == "__main__":
    main()
