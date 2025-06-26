import sys
sys.path.insert(1, '/home/PMO/airflow/dags')
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
import pandas as pd
from datetime import datetime
import Connections as conn
import ETLFunctions as fx
import pendulum
from pyspark.sql.functions import col
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

# MySQL connection string
load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"

# SQL Query
Receipts_2_Report_query = """ 
SELECT DISTINCT
    ACR.cash_receipt_id AS receipt_id,
    ACR.receipt_number,
    trx_head.org_id,
    ppa.name AS receipt_prj_name,
    acr.attribute1 AS receipt_prj_code,
    ACR.amount AS receipt_amount,
    acr.RECEIPT_DATE,
    trx_head.CUSTOMER_TRX_ID,
    TRX_HEAD.TRX_NUMBER AS Inv_num,
    ARA.AMOUNT_APPLIED,
    ARA.attribute1,
    COALESCE((SELECT SUM(AMOUNT) FROM AR_ADJUSTMENTS_ALL adj
              JOIN AR_RECEIVABLES_TRX_all RT ON ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
              WHERE TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
              AND adj.status NOT IN ('R', 'U')
              AND RT.NAME IN ('With Holding Tax')), 0) AS With_Holding_Tax,
    COALESCE((SELECT SUM(AMOUNT) FROM AR_ADJUSTMENTS_ALL adj
              JOIN AR_RECEIVABLES_TRX_all RT ON ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
              WHERE TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
              AND adj.status NOT IN ('R', 'U')
              AND RT.NAME IN ('Stamps', 'Stamp Tax')), 0) AS Stamp,
    TRX_HEAD.INVOICE_CURRENCY_CODE AS CURRENCY,
    (SELECT SUM(NVL(tl.QUANTITY_INVOICED, 0) * NVL(tl.UNIT_SELLING_PRICE, 1))
     FROM RA_CUSTOMER_TRX_LINES_ALL TL
     WHERE tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID) AS Transaction_Amount,
    (SELECT SUM(NVL(tax_amt, 0)) 
     FROM ZX_LINES_V 
     WHERE APPLICATION_ID = '222' 
     AND ENTITY_CODE = 'TRANSACTIONS' 
     AND EVENT_CLASS_CODE = 'INVOICE' 
     AND TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID 
     AND TRX_LEVEL_TYPE = 'LINE') AS Tax_Amount,
    (SELECT SUM(NVL(tl.QUANTITY_INVOICED, 0) * NVL(tl.UNIT_SELLING_PRICE, 1)) 
     FROM RA_CUSTOMER_TRX_LINES_ALL TL 
     WHERE tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID)
     + (SELECT SUM(NVL(tax_amt, 0)) 
        FROM ZX_LINES_V 
        WHERE APPLICATION_ID = '222' 
        AND ENTITY_CODE = 'TRANSACTIONS' 
        AND EVENT_CLASS_CODE = 'INVOICE' 
        AND TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID 
        AND TRX_LEVEL_TYPE = 'LINE') AS TOTAL_After_Tax,
    ARA.status,
    NVL(TRX_HEAD.EXCHANGE_RATE, 1) AS RATE,
    TO_CHAR(TRX_HEAD.TRX_DATE, 'DD-MM-YYYY') AS TRX_DATE,
    TO_CHAR(ARA.apply_date, 'DD-MM-YYYY') AS APPLY_DATE,
    TO_CHAR(TRX_HEAD.ATTRIBUTE2, 'YYYY/MM/DD HH24:MI:SS') AS DUE_DATE_DFF,
    hcaa.ACCOUNT_NUMBER AS Customer_No,
    hcaa.ACCOUNT_NAME AS Customer_Name
FROM AR_CASH_RECEIPTS_ALL ACR
JOIN pa_projects_all ppa ON ppa.segment1 = acr.attribute1
LEFT JOIN AR_RECEIVABLE_APPLICATIONS_ALL ARA ON acr.cash_receipt_id = ara.cash_receipt_id
LEFT JOIN RA_CUSTOMER_TRX_ALL TRX_HEAD ON TRX_HEAD.CUSTOMER_TRX_ID = ara.APPLIED_CUSTOMER_TRX_ID
LEFT JOIN HZ_CUST_ACCOUNTS HCAA ON HCAA.CUST_ACCOUNT_ID = TRX_HEAD.BILL_TO_CUSTOMER_ID
LEFT JOIN HR_OPERATING_UNITS HRO ON acr.org_id = hro.organization_id
WHERE acr.org_id = 83
GROUP BY 
    ACR.cash_receipt_id, ACR.receipt_number, trx_head.org_id, ppa.name, acr.attribute1, 
    ACR.amount, acr.RECEIPT_DATE, ara.apply_date, trx_head.CUSTOMER_TRX_ID, 
    ara.APPLIED_CUSTOMER_TRX_ID, ara.cash_receipt_id, TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1,
    ARA.AMOUNT_APPLIED, ara.attribute1, ara.status, TRX_HEAD.EXCHANGE_RATE, 
    TRX_HEAD.CUST_TRX_TYPE_ID, TRX_HEAD.TRX_NUMBER, TRX_HEAD.TRX_DATE, 
    TRX_HEAD.INVOICE_CURRENCY_CODE, TRX_HEAD.ATTRIBUTE2, hcaa.ACCOUNT_NUMBER, 
    hcaa.ACCOUNT_NAME, ara.APPLIED_PAYMENT_SCHEDULE_ID, TRX_HEAD.TERM_ID;
"""

# ETL Function
def Receipts_2_Report_ETL():
    try:
        spark = fx.spark_app('Receipts-2_Report', '2g', '2')
        print("Spark Session Initialized")
        
        RES = fx.connection(spark, 'RES', 'RMEDB', Receipts_2_Report_query, 'TEMP', 'ERP')
        
        if RES is not None:
            print("Query Execution Successful")

            # Ensure columns exist before casting
            for col_name in ["DUE_DATE_DFF", "APPLY_DATE", "TRX_DATE"]:
                if col_name in RES.columns:
                    RES = RES.withColumn(col_name, col(col_name).cast("string"))
            
            # Write to MySQL (Table Name Fixed)
            fx.WriteFunction(RES, load_connection_string, 'receipts_2', 'overwrite', conn.mysql_username, conn.mysql_password)    

            print("Data Written to MySQL Successfully")

    except Exception as e:
        print(f"Error Occurred: {e}")
        raise

# DAG Configuration
local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {
    'owner': 'sama',
    'start_date': datetime(2024, 11, 14, tzinfo=local_tz),
    "retries": 1,
}

dag = DAG(
    'Receipts-2_Report',
    catchup=False,
    default_args=default_args,
    schedule_interval='0 1 * * *',
    tags=['5']
)

Receipts_2_Report_Task = PythonOperator(
    dag=dag,
    task_id='Receipts-2_Report',
    python_callable=Receipts_2_Report_ETL
)

Receipts_2_Report_Task
