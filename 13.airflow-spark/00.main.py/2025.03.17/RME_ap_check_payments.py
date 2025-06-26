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
import mysql.connector as mysql

load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"


RME_ap_check_payments_query="""(
	SELECT
	ACA.CHECK_NUMBER "Document Number",
	ai.invoice_num,
	(
	SELECT
		DISTINCT xel.CODE_COMBINATION_ID
	FROM
		apps.AP_CHECKS_all ac,
		apps.xla_events xe,
		apps.xla_ae_headers xeh,
		apps.xla_ae_lines xel,
		xla.xla_transaction_entities xte
	WHERE
		xte.application_id = xe.application_id
		AND ac.CHECK_ID = aca.CHECK_ID
		AND xte.entity_code = 'AP_PAYMENTS'
		AND xte.source_id_int_1 = ac.CHECK_ID
		AND xte.entity_id = xe.entity_id
		AND xte.application_id = xeh.application_id
		AND xte.entity_id = xeh.entity_id
		AND xel.application_id = xeh.application_id
		AND xel.ae_header_id = xeh.ae_header_id
		AND ROWNUM = 1)
            comp_id,
	ai.attribute7,
	(
	SELECT
		DESCRIPTION
	FROM
		apps.FND_FLEX_VALUES_TL T,
		apps.FND_FLEX_VALUES B
	WHERE
		B.FLEX_VALUE_ID = T.FLEX_VALUE_ID
		AND FLEX_VALUE_SET_ID = 1017259
		AND FLEX_VALUE = ai.attribute7
		AND Description IS NULL)
            cost_center,
	ai.invoice_id,
	aip.amount PAYMENT_AMOUNT,
	aip.amount * NVL (aca.EXCHANGE_RATE,
	1) equiv,
	pap.segment1 project_number,
	pap.name project_name,
	NVL (
            (
	SELECT
		DISTINCT full_name
	FROM
		apps.pa_project_players pp,
		apps.per_all_people_f a
	WHERE
		pp.person_id = a.person_id
		AND pp.project_id = pap.project_id
		AND ROWNUM = 1
		AND PROJECT_ROLE_TYPE = '1000'),
	'General')
            owner,
	(
	SELECT
		DISTINCT SECTOR
	FROM
		apps.RME_PROJECT_SECTORS A,
		apps.pa_project_players pp,
		apps.per_all_people_f a
	WHERE
		1 = 1
		--PH.TRX_ID = RIH.TRX_ID
                            AND PAP.PROJECT_ID = A.PROJECT_ID
                            AND pp.person_id = a.person_id
                            AND pp.project_id = pap.project_id
                            AND ROWNUM = 1)SECTOR,
ASA.SEGMENT1 "Supplier Number",
  ASA.VENDOR_NAME "Supplier Name",
          --   ASA.SEGMENT1 "Supplier Num" ,
         aca.check_id,
         aca.CE_BANK_ACCT_USE_ID,
         cbac.BANK_ACCOUNT_ID,
         -- ASA.VENDOR_NAME "Sup Name",
         aca.VENDOR_ID,
         --    aca.BANK_ACCOUNT_ID ,
         aca.PAYCARD_AUTHORIZATION_NUMBER,
         aca.BANK_ACCOUNT_NAME,
         ACA.STATUS_LOOKUP_CODE "Payment Reconcilation Status",
         ACA.CLEARED_AMOUNT "Cleared Amount",
         ACA.CURRENCY_CODE "Currency",
         to_char(ACA.CLEARED_DATE,'YYYY-MM-DD') "Cleared Date",
         to_char(aca.CHECK_DATE,'YYYY-MM-DD') "CHECK_DATE",
         aca.amount,
       --  :p_vendor_site,
        --- :p_chk_from_date,
       --  :p_chk_to_date,
       --  :p_cle_from_date,
       --  :p_cle_to_date,
        -- :p_sup_num,
         CASE ASA.VENDOR_NAME
            WHEN 'Petty Cash'
            THEN
                  aca.ADDRESS_LINE1
               || DECODE (aca.ADDRESS_LINE1,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.ADDRESS_LINE2
               || DECODE (aca.ADDRESS_LINE2,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.ADDRESS_LINE3
               || DECODE (aca.ADDRESS_LINE3,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.CITY
               || ', '
               || aca.STATE
               || ' '
               || aca.ZIP
               || DECODE (
                     aca.CITY,
                     NULL, DECODE (
                              aca.STATE,
                              NULL, DECODE (aca.ZIP,
                                            NULL, '',
                                            APPS.fnd_global.local_chr (10)),
                              APPS.fnd_global.local_chr (10)),
                     APPS.fnd_global.local_chr (10))
               || aca.COUNTRY
            WHEN 'Staff Loan'
            THEN
                  aca.ADDRESS_LINE1
               || DECODE (aca.ADDRESS_LINE1,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.ADDRESS_LINE2
               || DECODE (aca.ADDRESS_LINE2,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.ADDRESS_LINE3
               || DECODE (aca.ADDRESS_LINE3,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.CITY
               || ', '
               || aca.STATE
               || ' '
               || aca.ZIP
               || DECODE (
                     aca.CITY,
                     NULL, DECODE (
                              aca.STATE,
                              NULL, DECODE (aca.ZIP,
                                            NULL, '',
                                            APPS.fnd_global.local_chr (10)),
                              APPS.fnd_global.local_chr (10)),
                     APPS.fnd_global.local_chr (10))
               || aca.COUNTRY
            WHEN 'RME Deposit To Other'
            THEN
                  aca.ADDRESS_LINE1
               || DECODE (aca.ADDRESS_LINE1,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.ADDRESS_LINE2
               || DECODE (aca.ADDRESS_LINE2,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.ADDRESS_LINE3
               || DECODE (aca.ADDRESS_LINE3,
                          NULL, '',
                          APPS.fnd_global.local_chr (10))
               || aca.CITY
               || ', '
               || aca.STATE
               || ' '
               || aca.ZIP
               || DECODE (
                     aca.CITY,
                     NULL, DECODE (
                              aca.STATE,
                              NULL, DECODE (aca.ZIP,
                                            NULL, '',
                                            APPS.fnd_global.local_chr (10)),
                              APPS.fnd_global.local_chr (10)),
                     APPS.fnd_global.local_chr (10))
               || aca.COUNTRY
            ELSE
               aca.VENDOR_SITE_CODE 
         END
            VENDOR_SITE_CODE,
         --     AIA.INVOICE_NUM,
         (SELECT DECODE (name,
                         'Alrowad Construction_OU', 'Rowad Modern Engineering',
                         name)
            FROM apps.hr_operating_units hou
           WHERE aca.org_id = hou.ORGANIZATION_ID)
            org_name,
         aca.org_id,
         --      AIA.INVOICE_AMOUNT,
         aca.STATUS_LOOKUP_CODE,
         TO_CHAR(aca.FUTURE_PAY_DUE_DATE,'YYYY-MM-DD') FUTURE_PAY_DUE_DATE,
         TO_CHAR(NVL (aca.FUTURE_PAY_DUE_DATE, aca.CHECK_DATE),'YYYY-MM-DD') maturaty_date,
         ai.attribute10 Category
    FROM APPS.AP_CHECKS_ALL ACA,                 --PAYMENT_METHOD,CHECK_NUMBER
         apps.ap_invoices_all ai,
         apps.ap_invoice_payments_all aip,
         --    APPS.AP_INVOICE_PAYMENTS_ALL AIPA,
         APPS.AP_SUPPLIERS ASA,
         apps.ce_bank_accounts cbac,
         apps.ce_bank_acct_uses_all cbau,
         apps.PA_PROJECTS_ALL PAP
   WHERE     aca.VENDOR_ID = ASA.VENDOR_ID
         AND cbau.BANK_ACCT_USE_ID = aca.CE_BANK_ACCT_USE_ID
         AND cbau.BANK_ACCOUNT_ID = cbac.BANK_ACCOUNT_ID
         AND ai.invoice_id(+) = aip.invoice_id
         --and ail.invoice_id = aip.invoice_id
         AND pap.project_id(+) = ai.ATTRIBUTE6
     --And pap.PROJECT_STATUS_CODE in 'APPROVED'
         --           AND ail.invoice_id = ai.invoice_id
         AND aip.check_id = aca.check_id
        and cbac.attribute1 != 'NON'
        and to_char(aca.CHECK_DATE,'YYYY-MM-DD') BETWEEN TO_CHAR(TRUNC(SYSDATE, 'YEAR'), 'YYYY-MM-DD') AND TO_CHAR(TRUNC(SYSDATE) - 1, 'YYYY-MM-DD')
)  temp """


def delete_data():
        db = mysql.connect(
         host = "10.10.11.242",
        user = conn.mysql_username,
        passwd = conn.mysql_password,database = "RME_TEST"
    )
        cursor = db.cursor()
        operation = "DELETE FROM RME_TEST.RME_ap_check_payments_Report WHERE CHECK_DATE between DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL DAYOFYEAR(CURDATE()) - 1 DAY), '%Y-%m-%d') AND DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 DAY), '%Y-%m-%d');"
        cursor.execute(operation)
        db.commit()

def RME_ap_check_payments_ETL():
    spark = fx.spark_app('RME_ap_check_payments','4g','4')
    RES = fx.connection(spark,'RES','RMEDB',RME_ap_check_payments_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'RME_ap_check_payments_Report','append',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,6, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('RME_ap_check_payments',catchup=False,default_args=default_args,schedule_interval='0 5 * * *',tags=['5'])


delete_dataTask= PythonOperator(dag=dag,
                task_id = 'delete_dataTask',
                python_callable=delete_data) 

ReadRME_ap_check_paymentsTask= PythonOperator(dag=dag,
                task_id = 'ReadRME_ap_check_paymentsTask',
                python_callable=RME_ap_check_payments_ETL) 


delete_dataTask >> ReadRME_ap_check_paymentsTask
