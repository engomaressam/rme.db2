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
mysql_table = "RME_Projects_Cost_Dist_Line_Report"

# Date range
START_YEAR = 2016
START_MONTH = 1

def get_columns_from_oracle():
    try:
        dsn = oracledb.makedsn(hostname, port, service_name=service_name)
        connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor_erp = connection_erp.cursor()
        cursor_erp.execute("SELECT * FROM RME_DEV.RME_PROJECTS_COST_DIST_LINE WHERE ROWNUM = 1")
        columns = [col[0] for col in cursor_erp.description]
        cursor_erp.close()
        connection_erp.close()
        return columns
    except Exception as e:
        print(f"❌ Oracle error (get_columns): {e}")
        return []

def ensure_table_exists(columns):
    if not columns:
        return
    try:
        mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        mysql_cursor = mysql_connection.cursor()
        col_defs = ", ".join([f"`{col}` TEXT" for col in columns])
        mysql_cursor.execute(f"CREATE TABLE IF NOT EXISTS {mysql_table} ({col_defs})")
        mysql_connection.commit()
        mysql_cursor.close()
        mysql_connection.close()
        print(f"✅ MySQL table {mysql_table} ensured with {len(columns)} columns.")
    except Exception as e:
        print(f"❌ MySQL error (ensure_table_exists): {e}")

def get_latest_month_in_mysql():
    try:
        mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        mysql_cursor = mysql_connection.cursor()
        mysql_cursor.execute(f"SELECT MAX(GL_DATE) FROM {mysql_table}")
        result = mysql_cursor.fetchone()[0]
        mysql_cursor.close()
        mysql_connection.close()
        if result is not None:
            return result.replace(day=1)
        else:
            return datetime(START_YEAR, START_MONTH, 1)
    except Exception as e:
        print(f"❌ MySQL error (get_latest_month): {e}")
        return datetime(START_YEAR, START_MONTH, 1)

def delete_month_from_mysql(year, month, columns):
    if not columns:
        return
    ensure_table_exists(columns)
    try:
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year+1, 1, 1)
        else:
            end_date = datetime(year, month+1, 1)
        mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
        mysql_cursor = mysql_connection.cursor()
        print(f"🗑️ Deleting rows for {start_date.strftime('%b-%Y')} from MySQL...")
        mysql_cursor.execute(f"DELETE FROM {mysql_table} WHERE GL_DATE >= %s AND GL_DATE < %s", (start_date, end_date))
        mysql_connection.commit()
        mysql_cursor.close()
        mysql_connection.close()
    except Exception as e:
        print(f"❌ MySQL error (delete_month): {e}")

def insert_to_mysql(df):
    if df is None or df.empty:
        return
    ensure_table_exists(df.columns)
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
            SELECT * FROM RME_DEV.RME_PROJECTS_COST_DIST_LINE
            WHERE GL_DATE >= :start_date AND GL_DATE < :end_date
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
    # Get columns and ensure MySQL table exists before starting
    columns = get_columns_from_oracle()
    ensure_table_exists(columns)
    # Find latest month in MySQL, or start from Jan 2016
    latest_date = get_latest_month_in_mysql()
    year = latest_date.year
    month = latest_date.month
    now = datetime.now()
    while (year < now.year) or (year == now.year and month <= now.month):
        print(f"\n=== Processing {datetime(year, month, 1).strftime('%b-%Y')} ===")
        df = fetch_month_from_oracle(year, month)
        if df is not None and not df.empty:
            print(f"⏳ DataFrame shape: {df.shape}. Preview:\n{df.head(3)}")
            delete_month_from_mysql(year, month, columns)
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
