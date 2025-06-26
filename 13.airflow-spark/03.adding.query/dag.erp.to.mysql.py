import sys
import os
import pendulum
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from airflow.exceptions import AirflowFailException
import mysql.connector as mysql
import oracledb
import pandas as pd

# ✅ Explicitly set the new non-Spark Oracle Client library path
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/oracle/21/client64_nonspark"
os.environ["ORACLE_HOME"] = "/usr/lib/oracle/21/client64_nonspark"
os.environ["TNS_ADMIN"] = "/usr/lib/oracle/21/client64_nonspark/network/admin"

# Import connection settings
import Connections as conn

# Retrieve MySQL credentials from Airflow Variables
mysql_host = Variable.get("MYSQL_HOST")
mysql_user = Variable.get("MYSQL_USER")
mysql_password = Variable.get("MYSQL_PASSWORD")
mysql_database = Variable.get("MYSQL_DATABASE")
mysql_table = "RME_Receipts_3"

# Define Oracle ERP Connection Details
oracle_hostname = "10.0.11.59"
oracle_port = 1521
oracle_service_name = "RMEDB"
oracle_username = "RME_DEV"
oracle_password = "PASS21RME"

def erp_to_mysql_etl():
    try:
        print("🔄 Setting up Oracle connection with non-Spark client...")
        
        # ✅ Use the new Oracle Client in Python
        oracledb.init_oracle_client(lib_dir="/usr/lib/oracle/21/client64_nonspark")

        print("🔄 Connecting to Oracle ERP...")
        dsn = oracledb.makedsn(oracle_hostname, oracle_port, service_name=oracle_service_name)
        connection_erp = oracledb.connect(user=oracle_username, password=oracle_password, dsn=dsn)
        cursor_erp = connection_erp.cursor()
        
        print("✅ Connected to Oracle ERP!")

        # ✅ Define query before executing
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
        print("🔄 Running query on Oracle ERP...")
        cursor_erp.execute(query_erp)
        data = cursor_erp.fetchall()
        columns = [desc[0].strip().upper() for desc in cursor_erp.description]
        df = pd.DataFrame(data, columns=columns)
        print(f"✅ Fetched {len(df)} rows from Oracle ERP.")

        # ✅ Convert dates & handle NULLs
        date_columns = ["RECEIPT_DATE"]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.date

        df = df.where(pd.notna(df), None).replace('', None)

        print("🔌 Connecting to MySQL...")
        mysql_connection = mysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_database)
        mysql_cursor = mysql_connection.cursor()

        print("🗑️ Dropping and recreating table in MySQL...")
        mysql_cursor.execute(f"DROP TABLE IF EXISTS {mysql_table}")
        create_table_query = f"""
        CREATE TABLE {mysql_table} (
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
            ATTRIBUTE1 VARCHAR(255)
        )
        """
        mysql_cursor.execute(create_table_query)
        mysql_connection.commit()

        print("📤 Inserting new data into MySQL table...")
        placeholders = ", ".join(["%s"] * len(df.columns))
        insert_query = f"INSERT INTO {mysql_table} ({', '.join(df.columns)}) VALUES ({placeholders})"
        mysql_cursor.executemany(insert_query, df.values.tolist())
        mysql_connection.commit()

        print(f"✅ Successfully inserted {len(df)} rows into MySQL.")

    except Exception as e:
        print(f"❌ Error during ETL: {e}")
        raise AirflowFailException(e)

    finally:
        if 'cursor_erp' in locals(): cursor_erp.close()
        if 'connection_erp' in locals(): connection_erp.close()
        if 'mysql_cursor' in locals(): mysql_cursor.close()
        if 'mysql_connection' in locals() and mysql_connection.is_connected(): mysql_connection.close()
        print("🔌 Connections closed.")

# Define Airflow DAG
local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {
    'owner': 'gamal',
    'start_date': datetime(2024, 11, 14, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
    'email': ['mohamed.Ghassan@rowad-rme.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}

dag = DAG(
    'erp_to_mysql_etl_nonspark',
    catchup=False,
    default_args=default_args,
    schedule_interval='30 5 * * *',
    tags=['ERP', 'MySQL', 'Non-Spark']
)

erp_to_mysql_task = PythonOperator(
    dag=dag,
    task_id='erp_to_mysql',
    python_callable=erp_to_mysql_etl
)

erp_to_mysql_task
