import sys
sys.path.insert(1, '/home/PMO/airflow/dags')
from pyspark.sql import SparkSession
from pyspark import SparkConf, conf
from pyspark.sql.types import *
import pandas as pd
from datetime import * 
import Connections as conn
import ETLFunctions as fx
from time import * 
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import random 
import sys 
import pendulum
from pyspark.sql.functions import to_date, col

load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"


Receipts_Report_query="""(
		/*Started by Ahmed Reda Ghanem*/
  SELECT DISTINCT
         ACR.cash_receipt_id receipt_id,
         ACR.receipt_number,
         ppa.name receipt_prj_name,
         acr.attribute1 receipt_prj_code,
         ACR.amount receipt_amount,
         acr.RECEIPT_DATE,
         trx_head.CUSTOMER_TRX_ID,
         TRX_HEAD.TRX_NUMBER Inv_num,
         ARA.AMOUNT_APPLIED,
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
           (SELECT SUM (
                        NVL (tl.QUANTITY_INVOICED, 0)
                      * NVL (tl.UNIT_SELLING_PRICE, 1))
              FROM RA_CUSTOMER_TRX_LINES_ALL TL
             WHERE tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID)
         + (SELECT SUM (NVL (tax_amt, 0)) Total_Tax
              FROM ZX_LINES_V
             WHERE     APPLICATION_ID = '222'
                   AND ENTITY_CODE = 'TRANSACTIONS'
                   AND EVENT_CLASS_CODE = 'INVOICE'
                   AND TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID
                   AND TRX_LEVEL_TYPE = 'LINE')
            TOTAL_After_Tax,
         (SELECT SUM (NVL (AMOUNT, 0))
            FROM AR_ADJUSTMENTS_ALL T
           WHERE     status NOT IN ('R', 'U')
                 AND (t.CUSTOMER_TRX_ID(+) = TRX_HEAD.CUSTOMER_TRX_ID))
            Adjustment_Amount,
           (SELECT SUM (
                        NVL (tl.QUANTITY_INVOICED, 0)
                      * NVL (tl.UNIT_SELLING_PRICE, 1))
              FROM RA_CUSTOMER_TRX_LINES_ALL TL
             WHERE tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID)
         + (SELECT SUM (NVL (tax_amt, 0)) Total_Tax
              FROM ZX_LINES_V
             WHERE     APPLICATION_ID = '222'
                   AND ENTITY_CODE = 'TRANSACTIONS'
                   AND EVENT_CLASS_CODE = 'INVOICE'
                   AND TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID
                   AND TRX_LEVEL_TYPE = 'LINE')
         + (SELECT SUM (NVL (AMOUNT, 0))
              FROM AR_ADJUSTMENTS_ALL T
             WHERE     status NOT IN ('R', 'U')
                   AND (t.CUSTOMER_TRX_ID(+) = TRX_HEAD.CUSTOMER_TRX_ID))
            Calculated_Amount_To_Collect,
         (SELECT SUM (NVL (arp.AMOUNT_APPLIED, 0))
            FROM AR_RECEIVABLE_APPLICATIONS_ALL arp,
                 ar_payment_schedules_all apsa,
                 RA_CUSTOMER_TRX_ALL rct
           WHERE     arp.APPLIED_PAYMENT_SCHEDULE_ID = apsa.PAYMENT_SCHEDULE_ID
                 AND rct.CUSTOMER_TRX_ID = apsa.CUSTOMER_trx_ID
                 AND rct.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID)
            Total_Amount_applied,
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
         (SELECT ppa.name
            FROM pa_projects_all ppa, ra_customer_trx_all trx_head
           WHERE     ppa.segment1 = trx_head.interface_header_attribute1
                 AND TRX_HEAD.CUSTOMER_TRX_ID = ara.APPLIED_CUSTOMER_TRX_ID
                 AND acr.cash_receipt_id = ara.cash_receipt_id
                 AND ROWNUM = 1)
            trx_prj_name,
         TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1 trx_Prj_code,
         NVL (
            (SELECT DISTINCT (F.FULL_NAME)
               FROM PA_PROJECTS_ALL PPA,
                    PA_PROJECT_PLAYERS PL,
                    PER_ALL_PEOPLE_F F
              WHERE     PPA.PROJECT_ID = PL.PROJECT_ID
                    AND PPA.SEGMENT1 = TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1
                    AND PL.PERSON_ID = F.PERSON_ID
                    AND PL.PROJECT_ROLE_TYPE = '1000'
                    AND ROWNUM = 1),
            'General')
            Project_Owner,
         ara.status,
         NVL (TRX_HEAD.EXCHANGE_RATE, 1) RATE,
         (SELECT TRX_LINE.QUANTITY_INVOICED
            FROM RA_CUSTOMER_TRX_ALL trx_head,
                 RA_CUSTOMER_TRX_LINES_ALL trx_line
           WHERE     trx_head.CUSTOMER_TRX_ID = trx_line.CUSTOMER_TRX_ID
                 AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
                 AND ROWNUM = 1)
            QTY,
         (SELECT ty.DESCRIPTION
            FROM RA_CUST_TRX_TYPES_ALL ty
           WHERE ty.CUST_TRX_TYPE_ID = TRX_HEAD.CUST_TRX_TYPE_ID AND ROWNUM = 1)
            DESCRIPTION,
         (SELECT ty.TYPE
            FROM RA_CUST_TRX_TYPES_ALL ty
           WHERE ty.CUST_TRX_TYPE_ID = TRX_HEAD.CUST_TRX_TYPE_ID AND ROWNUM = 1)
            TYPE,
         TO_CHAR (TRX_HEAD.TRX_DATE, 'DD-MM-YYYY') TRX_DATE,
         ara.apply_date,
         TO_DATE (TRX_HEAD.ATTRIBUTE2, 'YYYY/MM/DD HH24:MI:SS') DUE_DATE_DFF,
         (SELECT aps.PAYMENT_SCHEDULE_ID
            FROM ar_payment_schedules_all aps,
                 ra_customer_trx_all trx_head,
                 ra_customer_trx_lines_all trx_line
           WHERE     trx_head.customer_trx_id = trx_line.customer_trx_id(+)
                 AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
                 AND TRX_HEAD.customer_trx_id = aps.customer_trx_id(+)
                 AND TRX_HEAD.org_id = aps.org_id
                 AND ara.APPLIED_PAYMENT_SCHEDULE_ID = aps.PAYMENT_SCHEDULE_ID
                 AND ROWNUM = 1)
            PAYMENT_SCHEDULE_ID,
         hcaa.ACCOUNT_NUMBER Customer_No,
         hcaa.ACCOUNT_NAME Customer_Name
    ---------------------------------------------------------------------------------
    FROM AR_CASH_RECEIPTS_ALL acr,
         pa_projects_all ppa,
         AR_RECEIVABLE_APPLICATIONS_ALL ara,
         RA_CUSTOMER_TRX_ALL trx_head,
         RA_CUSTOMER_TRX_LINES_ALL trx_line,
         HZ_CUST_ACCOUNTS hcaa
   ---------------------------------------------------------------------------------
   WHERE     acr.cash_receipt_id = ara.cash_receipt_id(+)
         AND ppa.segment1 = acr.attribute1
         AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
         AND HCAA.CUST_ACCOUNT_ID(+) = trx_head.BILL_TO_CUSTOMER_ID
         AND ara.reversal_gl_date IS NULL
         AND ara.status IN ('ACC', 'APP')
---------------------------------------------------------------------------------
GROUP BY ACR.cash_receipt_id,
         ACR.receipt_number,
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
  --Ended by Ahmed Reda Ghanem
)  temp """

def Receipts_Report_ETL():
    spark = fx.spark_app('Receipts_Report', '2g', '2')
    RES = fx.connection(spark, 'RES', 'RMEDB', Receipts_Report_query, 'TEMP', 'ERP')
    
    if RES is not None:
        RES = RES.withColumn("DUE_DATE_DFF", to_date(col("DUE_DATE_DFF"), "yyyy-MM-dd"))
        fx.WriteFunction(RES, load_connection_string, 'Receipts_Report', 'overwrite', conn.mysql_username, conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {
    'owner': 'sama',
    'start_date': datetime(2024, 11, 14, tzinfo=local_tz),
    "retries": 1,
}

dag = DAG(
    'Receipts_Report',
    catchup=False,
    default_args=default_args,
    schedule_interval='0 1 * * *',
    tags=['5']
)

Receipts_Report_Task = PythonOperator(
    dag=dag,
    task_id='Receipts_Report',
    python_callable=Receipts_Report_ETL
)

Receipts_Report_Task

