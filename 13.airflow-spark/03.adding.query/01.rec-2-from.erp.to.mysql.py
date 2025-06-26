import os
import oracledb
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# ✅ Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# ✅ Database connection details
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

try:
    print("🔄 Connecting to the Oracle ERP database...")
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor_erp = connection_erp.cursor()
    print("✅ Successfully connected to Oracle ERP!")

except oracledb.Error as error:
    print(f"❌ Database connection error: {error}")

# ✅ Define the query with org_id = 83 hardcoded
query_erp = """
SELECT DISTINCT
         ACR.cash_receipt_id receipt_id,
         ACR.receipt_number,
         trx_head.org_id,
         ppa.name receipt_prj_name,
         acr.attribute1 receipt_prj_code,
         ACR.amount receipt_amount,
         acr.RECEIPT_DATE,
         trx_head.CUSTOMER_TRX_ID,
         TRX_HEAD.TRX_NUMBER Inv_num,
         ARA.AMOUNT_APPLIED,
         ARA.attribute1,
           ( select nvl(sum(AMOUNT),0)
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME in ('With Holding Tax')) With_Holding_Tax,
   
  ( select nvl(sum(AMOUNT),0) 
   FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
   where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
   and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
   and adj.status NOT IN ('R', 'U')
   and RT.NAME in ( 'Stamps','Stamp Tax')) Stamp,

   -- (Retention, Social, Tax, Other Conditions Logic remains unchanged)
   
         TRX_HEAD.INVOICE_CURRENCY_CODE CURRENCY,
         (SELECT SUM (
                      NVL (tl.QUANTITY_INVOICED, 0)
                    * NVL (tl.UNIT_SELLING_PRICE, 1))
            FROM RA_CUSTOMER_TRX_LINES_ALL TL
           WHERE tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID)
            Transaction_Amount,
         (SELECT NVL (SUM (extended_amount), 0)
            FROM ra_customer_trx_all trx_head,
                 ra_customer_trx_lines_all trx_line
           WHERE     trx_head.customer_trx_id = trx_line.customer_trx_id(+)
                 AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
                 AND trx_line.line_type IN ('TAX'))
            Tax_Amount,

         (SELECT aps.AMOUNT_DUE_REMAINING
            FROM ar_payment_schedules_all aps,
                 ra_customer_trx_all trx_head,
                 ra_customer_trx_lines_all trx_line
           WHERE     trx_head.customer_trx_id = trx_line.customer_trx_id(+)
                 AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
                 AND TRX_HEAD.customer_trx_id = aps.customer_trx_id(+)
                 AND TRX_HEAD.org_id = aps.org_id
                 AND ara.APPLIED_PAYMENT_SCHEDULE_ID = aps.PAYMENT_SCHEDULE_ID
                 AND ROWNUM = 1)
            AMOUNT_DUE_REMAINING,

         hcaa.ACCOUNT_NUMBER Customer_No,
         hcaa.ACCOUNT_NAME Customer_Name

FROM AR_CASH_RECEIPTS_ALL acr
LEFT JOIN pa_projects_all ppa ON ppa.segment1 = acr.attribute1
LEFT JOIN AR_RECEIVABLE_APPLICATIONS_ALL ara ON acr.cash_receipt_id = ara.cash_receipt_id
LEFT JOIN RA_CUSTOMER_TRX_ALL trx_head ON trx_head.CUSTOMER_TRX_ID = ara.APPLIED_CUSTOMER_TRX_ID
LEFT JOIN HZ_CUST_ACCOUNTS hcaa ON hcaa.CUST_ACCOUNT_ID = trx_head.BILL_TO_CUSTOMER_ID
LEFT JOIN hr_operating_units hro ON acr.org_id = hro.organization_id 

WHERE acr.org_id = 83  -- ✅ Hardcoded organization ID
AND ara.reversal_gl_date IS NULL
AND ara.status IN ('ACC', 'APP')

GROUP BY ACR.cash_receipt_id,
         ACR.receipt_number,
         trx_head.org_id,
         ppa.name,
         acr.attribute1,
         ACR.amount,
         acr.RECEIPT_DATE,
         ara.apply_date,
         trx_head.CUSTOMER_TRX_ID,
         ara.APPLIED_CUSTOMER_TRX_ID,
         ara.cash_receipt_id,
         TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1,
         ARA.AMOUNT_APPLIED,
         ara.attribute1,
         ara.status,
         TRX_HEAD.EXCHANGE_RATE,
         TRX_HEAD.CUST_TRX_TYPE_ID,
         TRX_HEAD.TRX_NUMBER,
         TRX_HEAD.TRX_DATE,
         TRX_HEAD.INVOICE_CURRENCY_CODE,
         TRX_HEAD.ATTRIBUTE2,
         hcaa.ACCOUNT_NUMBER,
         hcaa.ACCOUNT_NAME,
         ara.APPLIED_PAYMENT_SCHEDULE_ID,
         TRX_HEAD.TERM_ID
"""

# ✅ Confirm query is stored correctly
print("✅ Query stored successfully!")

import mysql.connector as mysql
from mysql.connector import Error

try:
    # ✅ Establish the MySQL connection
    print("🔄 Connecting to MySQL database...")
    mysql_connection = mysql.connect(
        host="10.10.11.242",
        user="omar2",
        password="Omar_54321",
        database="RME_TEST"
    )
    mysql_cursor = mysql_connection.cursor()
    
    if mysql_connection.is_connected():
        print("✅ Successfully connected to MySQL!")

except Error as e:
    print(f"❌ Error connecting to MySQL: {e}")

import pandas as pd
import mysql.connector as mysql
from mysql.connector import Error

try:
    # ✅ Execute the query from Cell 2 on Oracle ERP
    print("🔄 Running query on Oracle ERP...")
    cursor_erp.execute(query_erp)  # query_erp should be defined in Cell 2
    data = cursor_erp.fetchall()  # Fetch all results

    # ✅ Get column names
    columns = [desc[0].strip().upper() for desc in cursor_erp.description]  # Ensure uppercase, no spaces

    # ✅ Store results in a Pandas DataFrame
    df = pd.DataFrame(data, columns=columns)
    print(f"✅ Fetched {len(df)} rows from Oracle ERP.")

    # ✅ Print actual column names for debugging
    print(f"🧐 Columns in DataFrame: {df.columns.tolist()}")

    # ✅ Ensure all required columns exist, even if empty
    required_columns = [
        "RECEIPT_ID", "RECEIPT_NUMBER", "ORG_ID", "RECEIPT_PRJ_NAME", "RECEIPT_PRJ_CODE",
        "RECEIPT_AMOUNT", "RECEIPT_DATE", "CUSTOMER_TRX_ID", "INV_NUM", "AMOUNT_APPLIED",
        "ATTRIBUTE1", "WITH_HOLDING_TAX", "STAMP", "RETENTIONN", "SOCIAL", "TAX",
        "OTHER_CONDITIONS", "CURRENCY", "TRANSACTION_AMOUNT", "TAX_AMOUNT", "TRX_DATE",
        "APPLY_DATE", "DUE_DATE_DFF", "CUSTOMER_NO", "CUSTOMER_NAME", "AMOUNT_DUE_REMAINING"
    ]

    # ✅ Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' missing in DataFrame. Adding it with default NULL values.")
            df[col] = None  # Fill missing columns with NULL

    # ✅ Ensure column order matches MySQL
    df = df[required_columns]

    # ✅ Convert all NaN, empty strings, and invalid values to None
    df = df.where(pd.notna(df), None)

    # ✅ Check for any 'nan' values manually (Debugging)
    print("🔍 Checking for 'nan' values in DataFrame...")
    for col in df.columns:
        if df[col].isna().sum() > 0:
            print(f"⚠️ Column '{col}' contains {df[col].isna().sum()} NaN values. Replacing with NULL...")

    # ✅ Convert DataFrame to list of tuples for MySQL insertion
    data_tuples = [tuple(None if pd.isna(val) or val == "nan" or val == "" else val for val in row) for row in df.values]

    # ✅ Connect to MySQL
    print("🔌 Connecting to MySQL...")
    mysql_connection = mysql.connect(
        host="10.10.11.242",
        user="omar2",
        password="Omar_54321",
        database="RME_TEST"
    )
    mysql_cursor = mysql_connection.cursor()

    # ✅ Drop and recreate the table (to avoid schema mismatches)
    print("🗑️ Dropping and recreating RME_Receipts_3 table...")
    mysql_cursor.execute("DROP TABLE IF EXISTS RME_Receipts_3")
    
    create_table_query = """
    CREATE TABLE RME_Receipts_3 (
        RECEIPT_ID VARCHAR(50),
        RECEIPT_NUMBER VARCHAR(50),
        ORG_ID INT,
        RECEIPT_PRJ_NAME VARCHAR(255),
        RECEIPT_PRJ_CODE VARCHAR(255),
        RECEIPT_AMOUNT FLOAT,
        RECEIPT_DATE DATE,
        CUSTOMER_TRX_ID VARCHAR(50),
        INV_NUM VARCHAR(255),
        AMOUNT_APPLIED FLOAT,
        ATTRIBUTE1 VARCHAR(255),
        WITH_HOLDING_TAX FLOAT,
        STAMP FLOAT,
        RETENTIONN FLOAT DEFAULT NULL,
        SOCIAL FLOAT DEFAULT NULL,
        TAX FLOAT DEFAULT NULL,
        OTHER_CONDITIONS FLOAT DEFAULT NULL,
        CURRENCY VARCHAR(10),
        TRANSACTION_AMOUNT FLOAT,
        TAX_AMOUNT FLOAT,
        TRX_DATE DATE DEFAULT NULL,
        APPLY_DATE DATE DEFAULT NULL,
        DUE_DATE_DFF DATE DEFAULT NULL,
        CUSTOMER_NO VARCHAR(50),
        CUSTOMER_NAME VARCHAR(255),
        AMOUNT_DUE_REMAINING FLOAT DEFAULT NULL
    )
    """
    mysql_cursor.execute(create_table_query)
    mysql_connection.commit()

    # ✅ Insert new data into MySQL
    print("📤 Inserting new data into MySQL table RME_Receipts_3...")
    placeholders = ", ".join(["%s"] * len(df.columns))  # Create placeholders for values
    insert_query = f"INSERT INTO RME_Receipts_3 ({', '.join(df.columns)}) VALUES ({placeholders})"
    
    mysql_cursor.executemany(insert_query, data_tuples)
    mysql_connection.commit()

    print(f"✅ Successfully inserted {len(df)} rows into MySQL.")

except ValueError as ve:
    print(f"❌ Column Mismatch Error: {ve}")

except mysql.Error as mysql_err:
    print(f"❌ MySQL Error: {mysql_err}")

except oracledb.Error as oracle_err:
    print(f"❌ Oracle ERP Query Error: {oracle_err}")

except Exception as e:
    print(f"❌ Unexpected Error: {e}")

finally:
    # ✅ Close all connections safely
    if 'cursor_erp' in locals() and cursor_erp:
        cursor_erp.close()
    if 'connection' in locals() and connection:
        connection.close()
    if 'mysql_cursor' in locals() and mysql_cursor:
        mysql_cursor.close()
    if 'mysql_connection' in locals() and mysql_connection.is_connected():
        mysql_connection.close()
        print("🔌 MySQL connection closed.")
