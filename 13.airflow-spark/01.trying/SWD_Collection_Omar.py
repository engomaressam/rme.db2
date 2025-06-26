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
import sys 
import pendulum

load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"


SWD_Collection_query="""(
   SELECT 
	cash_receipt_id,
	receipt_number,
	to_char(receipt_date, 'YYYY-MM-DD') receipt_date,
	TYPE,
	amount,
	func_amount,
	unidentified_amount,
	applied_amount,
	on_account_amount,
	unapplied_amount,
	(unidentified_amount * conv_rate) unidentified_func_amount,
	(applied_amount * conv_rate) applied_func_amount,
	(on_account_amount * conv_rate) on_account_func_amount,
	(unapplied_amount * conv_rate) unapplied_func_amount,
	currency_code,
	STATUS,
	customer_id,
	activity_id,
	comments,
	rec_no,
	old_no,
	remit_bank_acct_use_id,
	receipt_method_id,
	payment_method ,
	project_num,
	project_name,
	owner
FROM
	(
	SELECT
		DISTINCT
                 cr.cash_receipt_id,
		cr.receipt_number,
		cr.receipt_date,
		DECODE (cr.TYPE,
		'CASH',
		'Standard',
		'Miscellaneous') TYPE,
		cr.amount,
		(cr.amount * NVL (cr.exchange_rate,
		1)) func_amount,
		(
		SELECT
			SUM (
                            NVL (APP.AMOUNT_APPLIED_FROM,
			APP.AMOUNT_APPLIED))
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status = 'UNID'
			AND app.cash_receipt_id = cr.cash_receipt_id
 )
                    unidentified_amount,
		(
		SELECT
			SUM (
                            NVL (APP.AMOUNT_APPLIED_FROM,
			APP.AMOUNT_APPLIED))
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status IN ('APP', 'ACTIVITY')
				AND app.cash_receipt_id = cr.cash_receipt_id
                 )
                    applied_amount,
		(
		SELECT
			SUM (
                            NVL (APP.AMOUNT_APPLIED_FROM,
			APP.AMOUNT_APPLIED))
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status = 'ACC'
			AND app.cash_receipt_id = cr.cash_receipt_id
                        
                                                    )
                    on_account_amount,
		(
		SELECT
			SUM (
                            NVL (APP.AMOUNT_APPLIED_FROM,
			APP.AMOUNT_APPLIED))
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status = 'UNAPP'
			AND app.cash_receipt_id = cr.cash_receipt_id
                                                    )
                    unapplied_amount,
		NVL (cr.exchange_rate,
		1) conv_rate,
		cr.currency_code,
		(
		SELECT
			OTR.STATUS
		FROM
			apps.AR_CASH_RECEIPT_HISTORY_ALL OTR
		WHERE
			OTR.CASH_RECEIPT_HISTORY_ID =
                            (
			SELECT
				MAX (INR.CASH_RECEIPT_HISTORY_ID)
			FROM
				apps.AR_CASH_RECEIPT_HISTORY_ALL INR
			WHERE
				INR.CASH_RECEIPT_ID =
                                           CR.CASH_RECEIPT_ID
                               ))
                    STATUS,
		party.party_id customer_id,
		cr.receivables_trx_id activity_id,
		cr.comments,
		cr.attribute1 rec_no,
		cr.attribute2 old_no,
		cr.remit_bank_acct_use_id,
		cr.receipt_method_id,
		rec_method.name payment_method ,
		cr.attribute1 project_num,
		pap.name project_name,
		nvl((SELECT DISTINCT full_name FROM apps.per_all_people_f paf , apps.pa_project_players ppp 
                  WHERE paf.person_id = ppp.person_id
                  AND pap.project_id = ppp.project_id
 AND SYSDATE BETWEEN paf.EFFECTIVE_START_DATE
                                     AND paf.EFFECTIVE_END_DATE
AND rownum = 1
                  AND PROJECT_ROLE_TYPE = '1000'), 'General') owner
	FROM
		apps.hz_cust_accounts cust,
		apps.hz_parties party,
		apps.ar_receipt_methods rec_method,
		apps.ar_cash_receipts_all cr,
		apps.PA_PROJECTS_ALL PAP ,
		apps.ar_cash_receipt_history_all crh_current,
		apps.AR_RECEIVABLE_APPLICATIONS_all APP,
		apps.AR_PAYMENT_SCHEDULES_all PS_INV,
		apps.RA_CUST_TRX_TYPES_all CTT
	WHERE
		cr.pay_from_customer = cust.cust_account_id(+)
		AND cust.party_id = party.party_id(+)
		AND crh_current.cash_receipt_id = cr.cash_receipt_id
		AND cr.receipt_method_id = rec_method.receipt_method_id
		AND cr.attribute1 = pap.segment1(+)
		AND crh_current.current_record_flag = 'Y'
		AND UPPER (
                        arpt_sql_func_util.get_lookup_meaning (
                           'RECEIPT_CREATION_STATUS',
		crh_current.status)) != 'REVERSED'
		AND APP.CASH_RECEIPT_ID = CR.CASH_RECEIPT_ID
		AND APP.APPLIED_PAYMENT_SCHEDULE_ID =
                        PS_INV.PAYMENT_SCHEDULE_ID(+)
		AND CTT.CUST_TRX_TYPE_ID(+) = PS_INV.CUST_TRX_TYPE_ID)
)  temp """


def SWD_Collection_Omar_ETL():
    spark = fx.spark_app('SWD_Collection_Omar','2g','2')
    RES = fx.connection(spark,'RES','RMEDB',SWD_Collection_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'SWD_Collection_Omar_Report','overwrite',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'omar2',
                'start_date' : datetime(2025,1,15, tzinfo=local_tz),"retries": 2,
                 "retry_delay": timedelta(minutes=40),'email': ['omar.essam@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('SWD_Collection_Omar',catchup=False,default_args=default_args,schedule_interval='5 2 * * *',tags=['2'])


SWD_Collection_Omar1Task= PythonOperator(dag=dag,
                task_id = 'SWD_Collection_Omar1',
                python_callable=SWD_Collection_Omar_ETL) 


SWD_Collection_Omar1Task
