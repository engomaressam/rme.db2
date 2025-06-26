import sys
sys.path.insert(1, '/home/PMO/airflow/dags')

from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
import pandas as pd
from datetime import datetime
import Connections as conn
import ETLFunctions as fx
from time import sleep
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pendulum
from pyspark.sql.functions import to_date, col

# MySQL connection
load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"

# Updated SQL query with correct schema (APPS)
Receipts_2_Report_query = """ 
SELECT DISTINCT
    ACR.cash_receipt_id AS receipt_id,
    ACR.receipt_number,
    trx_head.org_id,
    ppa.name AS receipt_prj_name,
    acr.attribute1 AS receipt_prj_code,
    ACR.amount AS receipt_amount,
    TO_CHAR(acr.RECEIPT_DATE, 'DD-MM-YYYY') AS RECEIPT_DATE,
    trx_head.CUSTOMER_TRX_ID,
    TRX_HEAD.TRX_NUMBER AS Inv_num,
    ARA.AMOUNT_APPLIED,
    ARA.attribute1
FROM APPS.AR_CASH_RECEIPTS_ALL ACR
JOIN APPS.pa_projects_all ppa ON ppa.segment1 = acr.attribute1
LEFT JOIN APPS.AR_RECEIVABLE_APPLICATIONS_ALL ARA ON acr.cash_receipt_id = ara.cash_receipt_id
LEFT JOIN APPS.RA_CUSTOMER_TRX_ALL TRX_HEAD ON TRX_HEAD.CUSTOMER_TRX_ID = ara.APPLIED_CUSTOMER_TRX_ID
LEFT JOIN APPS.HZ_CUST_ACCOUNTS HCAA ON HCAA.CUST_ACCOUNT_ID = TRX_HEAD.BILL_TO_CUSTOMER_ID
LEFT JOIN APPS.HR_OPERATING_UNITS HRO ON acr.org_id = hro.organization_id
WHERE acr.org_id = 83
"""

def Receipts_2_Report_ETL():
    spark = fx.spark_app('Receipts-2_Report', '2g', '2')
    RES = fx.connection(spark, 'RES', 'APPS', Receipts_2_Report_query, 'TEMP', 'ERP')

    if RES is not None:
        # Convert date columns to STRING
        RES = RES.withColumn("RECEIPT_DATE", col("RECEIPT_DATE").cast("string"))
        
        # Now write the transformed data to MySQL
        fx.WriteFunction(RES, load_connection_string, 'Receipts-2_Report', 'overwrite', conn.mysql_username, conn.mysql_password)

# DAG configuration
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
