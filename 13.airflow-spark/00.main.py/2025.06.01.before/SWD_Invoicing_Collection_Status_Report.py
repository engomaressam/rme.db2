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
   SELECT 
   TRX_HEAD.org_id,
   araa.CASH_RECEIPT_ID,
       TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1 PROJECT_NUMBER,
       ppa.NAME Project_name , 
       TRX_HEAD.CUSTOMER_TRX_ID,
       aps.PAYMENT_SCHEDULE_ID,
       araa.attribute1,
       sum(araa.AMOUNT_APPLIED) Amount_applied,
       (select sum(nvl(arp.AMOUNT_APPLIED, 0))
          from apps.AR_RECEIVABLE_APPLICATIONS_ALL arp,
               apps.ar_payment_schedules_all       apsa,
               apps.RA_CUSTOMER_TRX_ALL            rct
         where arp.APPLIED_PAYMENT_SCHEDULE_ID = apsa.PAYMENT_SCHEDULE_ID
           and rct.CUSTOMER_TRX_ID = apsa.CUSTOMER_trx_ID
           and rct.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID) Total_Amount_applied,
       hca.ACCOUNT_NUMBER Customer_No,
       hp.PARTY_NAME Customer_Name,
       cr.comments,
       TRX_HEAD.TRX_NUMBER Inv_num,
       (SELECT NVL(SUM(extended_amount), 0)
          FROM apps.ra_customer_trx_all       trx_head1,
               apps.ra_customer_trx_lines_all trx_line1
         WHERE trx_head1.customer_trx_id = trx_line1.customer_trx_id(+)
           AND trx_head.customer_trx_id = trx_head1.customer_trx_id
           AND trx_line1.customer_trx_id = trx_line.customer_trx_id
           AND trx_head1.org_id = trx_line1.org_id(+)
           AND trx_line1.line_type IN ('TAX')) VAT,
       ty.DESCRIPTION Inv_Type,
       --Started By A.Zaki 08-10-2017 to use it when calculate AR balance invoice
       ty.TYPE INV_CLASS,
       --Ended By A.Zaki 08-10-2017
       TO_CHAR(TRX_HEAD.TRX_DATE, 'YYYY-MM-DD') Invoice_DATE,
       TRX_HEAD.INVOICE_CURRENCY_CODE CURRENCY,
       NVL(TRX_HEAD.EXCHANGE_RATE, 1) RATE,
       aps.AMOUNT_DUE_REMAINING,
       --ROUND (TRX_LINE.UNIT_SELLING_PRICE, 4) UNIT_SELLING_PRICE,
       (select SUM(NVL(tl.QUANTITY_INVOICED, 0) *
                   NVL(tl.UNIT_SELLING_PRICE, 1))
          from apps.RA_CUSTOMER_TRX_LINES_ALL TL
         where tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID) Amount_Execluded_Vat,
       
       ----------------------------------------------
       NVL((select SUM(NVL(tl.QUANTITY_INVOICED, 0) *
                   NVL(tl.UNIT_SELLING_PRICE, 1))
          from apps.RA_CUSTOMER_TRX_LINES_ALL TL
         where tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID) +
      (SELECT sum(nvl(tax_amt, 0)) Total_Tax
          FROM apps.ZX_LINES_V
         WHERE APPLICATION_ID = '222'
           AND ENTITY_CODE = 'TRANSACTIONS'
           AND EVENT_CLASS_CODE = 'INVOICE'
           AND TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID
           AND TRX_LEVEL_TYPE = 'LINE'),0) Amount_Included_Vat, 
          nvl(lookup.With_Holding_Tax,0) With_Holding_Tax,
          nvl(lookup.Stamps,0) Stamps,
          nvl(lookup.Retention,0) Retention,
          nvl(lookup.Social,0) Social,
          nvl(lookup.Material_On_Site,0) Material_On_Site,
          nvl(lookup.Other_in_System,0) Other_in_System,
          nvl(lookup.Other_Conditions,0) Other_Conditions,
       NVL((SELECT sum(nvl(AMOUNT, 0))
          FROM apps.AR_ADJUSTMENTS_ALL T
         WHERE status not in ('R', 'U')
           and (t.CUSTOMER_TRX_ID(+) = TRX_HEAD.CUSTOMER_TRX_ID)),0) Total_Adjustment_Amount,
       ----------------------------------------------
       
       --SUM(TRX_LINE.QUANTITY_INVOICED) QTY,
       /*ARPT_SQL_FUNC_UTIL.GET_FIRST_REAL_DUE_DATE(TRX_HEAD.CUSTOMER_TRX_ID,
                                                  TRX_HEAD.TERM_ID,
                                                  TRX_HEAD.TRX_DATE) TERM_DUE_DATE,*/
       TO_CHAR(TO_DATE (TRX_HEAD.ATTRIBUTE2, 'YYYY/MM/DD HH24:MI:SS'),'YYYY-MM-DD') DUE_DATE_DFF,
       NVL((SELECT DISTINCT (F.FULL_NAME)
             FROM apps.PA_PROJECTS_ALL    PPA,
                  apps.PA_PROJECT_PLAYERS PL,
                  apps.PER_ALL_PEOPLE_F   F
            WHERE PPA.PROJECT_ID = PL.PROJECT_ID
              AND PPA.SEGMENT1 = TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1
              AND PL.PERSON_ID = F.PERSON_ID
              AND PL.PROJECT_ROLE_TYPE = '1000'
                 
              AND ROWNUM = 1),
           'General') Project_Owner
  FROM apps.RA_CUSTOMER_TRX_ALL            TRX_HEAD,
       apps.RA_CUSTOMER_TRX_LINES_ALL      TRX_LINE,
       apps.hz_parties                     hp,
       apps.HZ_CUST_ACCOUNTS               hca,
       apps.HZ_CUST_SITE_USES_ALL          hcsua,
       apps.HZ_CUST_ACCT_SITES_ALL         hcasa,
       apps.hz_party_sites                 hps,
       apps.RA_CUST_TRX_TYPES_ALL          ty,
       apps.ar_payment_schedules_all       aps,
       apps.AR_RECEIVABLE_APPLICATIONS_ALL araa ,
       apps.pa_projects_all    ppa ,
       apps.ar_cash_receipts_all cr,
       (
       SELECT
			    a.customer_trx_id,
			    NVL(SUM(CASE WHEN b.name IN ('W Holding Tax', 'With Holding Tax', 'WithHolding Tax', 'Withholding Tax') THEN NVL(a.amount, 0) END), 0) AS With_Holding_Tax,
			    NVL(SUM(CASE WHEN b.name IN ('Stamp Tax', 'Stamps', 'Stamps 2', 'Stamps Tax') THEN NVL(a.amount, 0) END), 0) AS Stamps,
			    NVL(SUM(CASE WHEN b.name LIKE '%Retention%' THEN NVL(a.amount, 0) END), 0) AS Retention,
			    NVL(SUM(CASE WHEN b.name IN ('Social Insurans', 'Social Insurans 2', 'Social insurance') THEN NVL(a.amount, 0) END), 0) AS Social,
			    NVL(SUM(CASE WHEN b.name = 'Material On Site' THEN NVL(a.amount, 0) END), 0) AS Material_On_Site,
			    NVL(SUM(CASE WHEN b.name LIKE '%Other%' THEN NVL(a.amount, 0) END), 0) AS Other_in_System,
			    NVL(SUM(CASE WHEN b.name NOT IN (
			            'W Holding Tax', 'With Holding Tax', 'WithHolding Tax', 'Withholding Tax',
			            'Stamp Tax', 'Stamps', 'Stamps 2', 'Stamps Tax',
			            'Social Insurans', 'Social Insurans 2', 'Social insurance',
			            'Material On Site'
			        ) AND b.name NOT LIKE '%Retention%'
			          AND b.name NOT LIKE '%Other%'
			        THEN NVL(a.AMOUNT, 0) END), 0) AS Other_Conditions
			FROM
			    apps.AR_ADJUSTMENTS_ALL a
			LEFT JOIN apps.AR_RECEIVABLES_TRX_ALL b ON
			    a.receivables_trx_id = b.receivables_trx_id
			WHERE
			    a.status NOT IN ('R', 'U')
			    --AND a.customer_trx_id = 1304583
			GROUP BY
			    a.customer_trx_id
       ) lookup
 WHERE TRX_HEAD.CUSTOMER_TRX_ID = TRX_LINE.CUSTOMER_TRX_ID
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
   AND TRX_HEAD.customer_trx_id = aps.customer_trx_id(+)
   AND TRX_HEAD.org_id = aps.org_id
   and araa.APPLIED_PAYMENT_SCHEDULE_ID(+) = aps.PAYMENT_SCHEDULE_ID
    AND trx_head.INTERFACE_HEADER_ATTRIBUTE1=ppa.segment1(+)   
      --AND TRX_HEAD.CUSTOMER_TRX_ID = araa.APPLIED_CUSTOMER_TRX_ID
    AND cr.cash_receipt_id(+)=araa.CASH_RECEIPT_ID
   AND DECODE(TRX_HEAD.SOLD_TO_CUSTOMER_ID,
              NULL,
              BILL_TO_CUSTOMER_ID,
              TRX_HEAD.SOLD_TO_CUSTOMER_ID) = hca.CUST_ACCOUNT_ID
   AND lookup.customer_trx_id(+)=TRX_HEAD.CUSTOMER_TRX_ID     
       --and araa.cash_receipt_id = 1967115
   --and TRX_HEAD.CUSTOMER_TRX_ID = 265774
   --and TRX_HEAD.org_id=83
-------------------------END USER PARAMETER-------------------------------------------------------------------------------
GROUP BY 
TRX_HEAD.org_id,
TRX_HEAD.CUSTOMER_TRX_ID,
cr.comments,
          TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1 ,
          ppa.NAME,
          araa.CASH_RECEIPT_ID,  
          aps.PAYMENT_SCHEDULE_ID,
          araa.attribute1,
          hca.ACCOUNT_NAME,
          PARTY_NAME,
          TO_CHAR(TRX_HEAD.TRX_DATE, 'YYYY-MM-DD') ,
          TRX_LINE.CUSTOMER_TRX_ID,
          TRX_HEAD.TRX_NUMBER,
          ty.DESCRIPTION,
          --Started By A.Zaki 08-10-2017 to use it when calculate AR balance invoice
          ty.TYPE,
          --Ended By A.Zaki 08-10-2017
          TRX_HEAD.INVOICE_CURRENCY_CODE,
          NVL(TRX_HEAD.EXCHANGE_RATE, 1),
          'S_TAX',
          --  TRX_LINE.UNIT_SELLING_PRICE,
          hca.ACCOUNT_NUMBER,
          aps.AMOUNT_DUE_REMAINING,
          /*ARPT_SQL_FUNC_UTIL.GET_FIRST_REAL_DUE_DATE(TRX_HEAD.CUSTOMER_TRX_ID,
                                                     TRX_HEAD.TERM_ID,
                                                     TRX_HEAD.TRX_DATE),*/
          TO_CHAR(TO_DATE (TRX_HEAD.ATTRIBUTE2, 'YYYY/MM/DD HH24:MI:SS'),'YYYY-MM-DD') ,
          lookup.With_Holding_Tax,
          lookup.Stamps,
          lookup.Retention,
          lookup.Social,
          lookup.Material_On_Site,
          lookup.Other_in_System,
          lookup.Other_Conditions
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
dag = DAG('SWD_Invoicing_Collection_Status',catchup=False,default_args=default_args,schedule_interval='0 7 * * *',tags=['3'])


SWD_Invoicing_Collection_StatusTask= PythonOperator(dag=dag,
                task_id = 'SWD_Invoicing_Collection_Status',
                python_callable=SWD_Invoicing_Collection_Status_ETL) 


SWD_Invoicing_Collection_StatusTask
