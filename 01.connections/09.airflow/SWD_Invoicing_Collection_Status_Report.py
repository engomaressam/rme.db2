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

load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"


SWD_Invoicing_Collection_Status_query="""(
   SELECT araa.CASH_RECEIPT_ID , 
         TRX_HEAD.CUSTOMER_TRX_ID,
         aps.PAYMENT_SCHEDULE_ID ,
         sum(DISTINCT araa.AMOUNT_APPLIED) Collected_Amount , 
         hca.ACCOUNT_NUMBER Customer_No,
         hp.PARTY_NAME Customer_Name,
         TRX_HEAD.TRX_NUMBER Inv_num,
         (SELECT NVL(SUM( DISTINCT extended_amount), 0)
            FROM APPS.ra_customer_trx_all trx_head1,
                 APPS.ra_customer_trx_lines_all trx_line1
           WHERE     trx_head1.customer_trx_id = trx_line1.customer_trx_id(+)
                 AND trx_head.customer_trx_id = trx_head1.customer_trx_id
                 AND trx_line1.customer_trx_id = trx_line.customer_trx_id
                 AND trx_head1.org_id = trx_line1.org_id(+)
                 AND trx_line1.line_type IN ('TAX'))
            VAT,
         ty.DESCRIPTION Inv_Type,
         --Started By A.Zaki 08-10-2017 to use it when calculate AR balance invoice
         ty.TYPE INV_CLASS,
         --Ended By A.Zaki 08-10-2017
         TO_CHAR(TRX_HEAD.TRX_DATE, 'YYYY-MM-DD') Invoice_DATE,
         TRX_HEAD.INVOICE_CURRENCY_CODE CURRENCY,
         NVL (TRX_HEAD.EXCHANGE_RATE, 1) RATE,
         aps.AMOUNT_DUE_REMAINING,
         --ROUND (TRX_LINE.UNIT_SELLING_PRICE, 4) UNIT_SELLING_PRICE,
         SUM (DISTINCT NVL(TRX_LINE.EXTENDED_AMOUNT, 0)) AMOUNT_EXECLUDED_VAT,
           SUM ( DISTINCT NVL(TRX_LINE.EXTENDED_AMOUNT, 0))
         + (SELECT NVL(SUM(DISTINCT extended_amount), 0)
            FROM APPS.ra_customer_trx_all trx_head1,
                 APPS.ra_customer_trx_lines_all trx_line1
           WHERE     trx_head1.customer_trx_id = trx_line1.customer_trx_id(+)
                 AND trx_head.customer_trx_id = trx_head1.customer_trx_id
                 AND trx_line1.customer_trx_id = trx_line.customer_trx_id
                 AND trx_head1.org_id = trx_line1.org_id(+)
                 AND trx_line1.line_type IN ('TAX'))
            AMOUNT_INCLUDED_VAT,
         SUM(DISTINCT TRX_LINE.QUANTITY_INVOICED) QTY,
         trx_head.INTERFACE_HEADER_ATTRIBUTE1 PROJECT_NUMBER,
         p.name PROJECT_NAME,
         /*APPS.ARPT_SQL_FUNC_UTIL.Get_First_Real_Due_Date(TRX_HEAD.CUSTOMER_TRX_ID,
                                                     TRX_HEAD.TERM_ID,
                                                     TRX_HEAD.TRX_DATE)
            TERM_DUE_DATE,*/
         TO_CHAR(TO_DATE (TRX_HEAD.ATTRIBUTE2, 'YYYY/MM/DD HH24:MI:SS'),'YYYY-MM-DD')  DUE_DATE_DFF,
         NVL (
            (SELECT DISTINCT (F.FULL_NAME)
               FROM APPS.PA_PROJECTS_ALL PPA,
                    APPS.PA_PROJECT_PLAYERS PL,
                    APPS.PER_ALL_PEOPLE_F F
              WHERE     PPA.PROJECT_ID = PL.PROJECT_ID
                    AND PPA.SEGMENT1 = TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1
                    AND PL.PERSON_ID = F.PERSON_ID
                    AND PL.PROJECT_ROLE_TYPE = '1000'
                    
                    AND ROWNUM = 1),
            'General')
            Project_Owner
    FROM APPS.RA_CUSTOMER_TRX_ALL TRX_HEAD,
         APPS.RA_CUSTOMER_TRX_LINES_ALL TRX_LINE,
         APPS.hz_parties hp,
         APPS.HZ_CUST_ACCOUNTS hca,
         APPS.HZ_CUST_SITE_USES_ALL hcsua,
         APPS.HZ_CUST_ACCT_SITES_ALL hcasa,
         APPS.hz_party_sites hps,
         APPS.RA_CUST_TRX_TYPES_ALL ty, 
         APPS.ar_payment_schedules_all aps , 
         APPS.AR_RECEIVABLE_APPLICATIONS_ALL araa,
         APPS.pa_projects_all  p 
   WHERE     TRX_HEAD.CUSTOMER_TRX_ID = TRX_LINE.CUSTOMER_TRX_ID
         AND TRX_HEAD.ORG_ID = TRX_LINE.ORG_ID(+)
         AND HCA.PARTY_ID = HP.PARTY_ID
         AND TRX_HEAD.COMPLETE_FLAG = 'Y'
         AND hps.status = 'A'
         --   AND TAX.LINK_TO_CUST_TRX_LINE_ID(+) = TRX_LINE.CUSTOMER_TRX_LINE_ID
         AND TRX_LINE.LINE_TYPE IN ('LINE', 'FREIGHT')
         AND IDENTIFYING_ADDRESS_FLAG = 'Y'
        AND hcsua.org_id = TRX_HEAD.org_id
         AND hcsua.site_use_code = 'BILL_TO'
         AND hcsua.cust_acct_site_id = hcasa.cust_acct_site_id
         AND hp.party_id = hps.party_id
         AND hca.CUST_ACCOUNT_ID = hcasa.CUST_ACCOUNT_ID
         AND hcsua.PRIMARY_FLAG = 'Y'
         AND ty.CUST_TRX_TYPE_ID = TRX_HEAD.CUST_TRX_TYPE_ID
         AND ty.org_id = TRX_HEAD.org_id
         AND TRX_HEAD.customer_trx_id = aps.customer_trx_id
         AND TRX_HEAD.org_id = aps.org_id
         and araa.APPLIED_PAYMENT_SCHEDULE_ID  = aps.PAYMENT_SCHEDULE_ID
         AND trx_head.INTERFACE_HEADER_ATTRIBUTE1=p.segment1(+)
         --AND TRX_HEAD.CUSTOMER_TRX_ID = araa.APPLIED_CUSTOMER_TRX_ID
         AND DECODE (TRX_HEAD.SOLD_TO_CUSTOMER_ID,
                     NULL, BILL_TO_CUSTOMER_ID,
                     TRX_HEAD.SOLD_TO_CUSTOMER_ID) = hca.CUST_ACCOUNT_ID
                     
        --and araa.cash_receipt_id = 1967115
        -- AND  trx_head.INTERFACE_HEADER_ATTRIBUTE1 ='0172' AND TRX_HEAD.TRX_NUMBER ='Rolling Mill ipc 14'

        

-------------------------END USER PARAMETER-------------------------------------------------------------------------------
GROUP BY TRX_HEAD.CUSTOMER_TRX_ID,
         araa.CASH_RECEIPT_ID , 
         aps.PAYMENT_SCHEDULE_ID ,
         hca.ACCOUNT_NAME,
         PARTY_NAME,
         TRX_HEAD.TRX_DATE,
         TRX_LINE.CUSTOMER_TRX_ID,
         TRX_HEAD.TRX_NUMBER,
         ty.DESCRIPTION,
         --Started By A.Zaki 08-10-2017 to use it when calculate AR balance invoice
         ty.TYPE,
         --Ended By A.Zaki 08-10-2017
         TRX_HEAD.INVOICE_CURRENCY_CODE,
         NVL (TRX_HEAD.EXCHANGE_RATE, 1),
         'S_TAX',
         --  TRX_LINE.UNIT_SELLING_PRICE,
         hca.ACCOUNT_NUMBER,
         trx_head.INTERFACE_HEADER_ATTRIBUTE1,
         p.name ,
         aps.AMOUNT_DUE_REMAINING,
         /*APPS.ARPT_SQL_FUNC_UTIL.Get_First_Real_Due_Date(
            TRX_HEAD.CUSTOMER_TRX_ID,
            TRX_HEAD.TERM_ID,
            TRX_HEAD.TRX_DATE),*/
         TO_CHAR(TO_DATE (TRX_HEAD.ATTRIBUTE2, 'YYYY/MM/DD HH24:MI:SS'),'YYYY-MM-DD') 
)  temp """


def SWD_Invoicing_Collection_Status_ETL():
    spark = fx.spark_app('SWD_Invoicing_Collection_Status','4g','4')
    RES = fx.connection(spark,'RES','RMEDB',SWD_Invoicing_Collection_Status_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'SWD_Invoicing_Collection_Status_Report','overwrite',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('SWD_Invoicing_Collection_Status',catchup=False,default_args=default_args,schedule_interval='0 3 * * *',tags=['3'])


SWD_Invoicing_Collection_StatusTask= PythonOperator(dag=dag,
                task_id = 'SWD_Invoicing_Collection_Status',
                python_callable=SWD_Invoicing_Collection_Status_ETL) 


SWD_Invoicing_Collection_StatusTask
