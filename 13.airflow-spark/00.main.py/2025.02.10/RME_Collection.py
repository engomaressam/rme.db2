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


RME_Collection_query="""(
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
	payment_method,
	project_num,
	project_name,
	owner
FROM
	(
	SELECT
		DISTINCT cr.cash_receipt_id,
		cr.receipt_number,
		cr.receipt_date,
		DECODE(cr.TYPE, 'CASH', 'Standard', 'Miscellaneous') TYPE,
		cr.amount,
		(cr.amount * NVL(cr.exchange_rate, 1)) func_amount,
		(
		SELECT
			SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED))
			--nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status = 'UNID'
			AND app.cash_receipt_id = cr.cash_receipt_id
			-- start by mohamed.dagher 12-02-2015 to include show amount by history
			---shark     
			--  And (TRUNC(app.APPLY_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
			--  And (TRUNC(app.APPLY_DATE) <= TRUNC(:p_to_date)   Or :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
			--- --SHARK 
                         ) unidentified_amount,
		(
		SELECT
			SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED))
			--nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
			-- start by mohamed.dagher 31-05-2016 to filter by bu
			--Started By A.Zaki 29-01-2018
			--,RA_CUST_TRX_TYPES_all CTI
			--Ended By A.Zaki 29-01-2018
			-- end by mohamed.dagher
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status IN ('APP', 'ACTIVITY')
				AND app.cash_receipt_id = cr.cash_receipt_id
				-- start by mohamed.dagher 12-02-2015 to include show amount by history
				--  And (TRUNC(app.APPLY_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
				--  And (TRUNC(app.APPLY_DATE) <= TRUNC(:p_to_date)   Or :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
				-- start by mohamed.dagher 12-02-2015 to include show amount by history
				--Started By A.Zaki 29-01-2018
				--And CTI.CUST_TRX_TYPE_ID(+) = PS_INV.CUST_TRX_TYPE_ID
				--And (ctI.attribute2 = :p_bu Or :p_bu Is Null)-- end by mohamed.dagher 12-02-2015 to include show amount by history
				--Ended By A.Zaki 29-01-2018
                         ) applied_amount,
		(
		SELECT
			SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED))
			--nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status = 'ACC'
			AND app.cash_receipt_id = cr.cash_receipt_id
			-- start by mohamed.dagher 12-02-2015 to include show amount by history
			---shark  
			--   And (TRUNC(app.gl_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
			--   And (TRUNC(app.gl_DATE) <= TRUNC(:p_to_date) Or   :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
			---shark  
                         ) on_account_amount,
		(
		SELECT
			SUM(NVL(APP.AMOUNT_APPLIED_FROM,
                                        APP.AMOUNT_APPLIED))
			--nvl(sum( APP.AMOUNT_APPLIED * GL_CURRENCY_API.GET_RATE(NVL(PS_INV.INVOICE_CURRENCY_CODE, CR.CURRENCY_CODE), CR.CURRENCY_CODE, APP.APPLY_DATE, 'Corporate') ), 0)
		FROM
			apps.ar_receivable_applications_all app,
			apps.AR_PAYMENT_SCHEDULES_ALL PS_INV
		WHERE
			APP.APPLIED_PAYMENT_SCHEDULE_ID =
                                PS_INV.PAYMENT_SCHEDULE_ID(+)
			AND app.status = 'UNAPP'
			AND app.cash_receipt_id = cr.cash_receipt_id
			-- start by mohamed.dagher 12-02-2015 to include show amount by history
			-----shark     
			-- And (TRUNC(app.APPLY_DATE) >= TRUNC(:p_from_date) Or :p_from_date Is Null)
			-- And (TRUNC(app.APPLY_DATE) <= TRUNC(:p_to_date) Or   :p_to_date Is Null) -- end by mohamed.dagher 12-02-2015
                         ) unapplied_amount,
		NVL(cr.exchange_rate, 1) conv_rate,
		cr.currency_code,
		--arpt_sql_func_util.get_lookup_meaning('RECEIPT_CREATION_STATUS', crh_current.status) status,
                        
                        (
		SELECT
			OTR.STATUS
		FROM
			apps.AR_CASH_RECEIPT_HISTORY_ALL OTR
		WHERE
			OTR.CASH_RECEIPT_HISTORY_ID =
                                (
			SELECT
				MAX(INR.CASH_RECEIPT_HISTORY_ID)
			FROM
				apps.AR_CASH_RECEIPT_HISTORY_ALL INR
			WHERE
				INR.CASH_RECEIPT_ID =
                                        CR.CASH_RECEIPT_ID
				--- SHARK 
				--And (INR.GL_DATE >= :P_FROM_DATE Or    :P_FROM_DATE Is Null)
				--And (INR.GL_DATE <= :P_TO_DATE   Or    :P_TO_DATE Is Null)
                                    )
                                    ) STATUS,
		party.party_id customer_id,
		cr.receivables_trx_id activity_id,
		cr.comments,
		cr.attribute1 rec_no,
		cr.attribute2 old_no,
		cr.remit_bank_acct_use_id,
		cr.receipt_method_id,
		rec_method.name payment_method,
		cr.attribute1 project_num,
		pap.name project_name,
		nvl((SELECT DISTINCT full_name
                              FROM apps.per_all_people_f paf,
                                   apps.pa_project_players ppp
                             WHERE paf.person_id = ppp.person_id
                               AND pap.project_id = ppp.project_id
                               AND SYSDATE BETWEEN paf.EFFECTIVE_START_DATE AND
                                   paf.EFFECTIVE_END_DATE
                               AND rownum = 1
                               AND PROJECT_ROLE_TYPE = '1000'),
                            'General') owner
	FROM
		apps.hz_cust_accounts cust,
		apps.hz_parties party,
		apps.ar_receipt_methods rec_method,
		apps.ar_cash_receipts_all cr,
		apps.PA_PROJECTS_ALL PAP,
		apps.ar_cash_receipt_history_all crh_current
		-- START BY MOHAMED.DAGHER 31-05-2016 TO FILTER BY BUSINESS UNTI
              ,
		apps.AR_RECEIVABLE_APPLICATIONS_all APP,
		apps.AR_PAYMENT_SCHEDULES_all PS_INV,
		apps.RA_CUST_TRX_TYPES_all CTT
		-- END BY MOHAMED.DAGHER              --,
        /*
        (select cash_receipt_id, status, sum(amount_applied) app_amount
         from   ar_receivable_applications_all app
         group by cash_receipt_id, status) receipt_apply
                   */
	WHERE
		cr.pay_from_customer = cust.cust_account_id(+)
		AND cust.party_id = party.party_id(+)
		AND crh_current.cash_receipt_id = cr.cash_receipt_id
		AND cr.receipt_method_id = rec_method.receipt_method_id
		--AND    receipt_apply.cash_receipt_id = cr.cash_receipt_id
		AND cr.attribute1 = pap.segment1(+)
		AND crh_current.current_record_flag = 'Y'
		AND UPPER(APPS.ARPT_SQL_FUNC_UTIL.GET_LOOKUP_MEANING('RECEIPT_CREATION_STATUS', crh_current.status)) != 'REVERSED'
		--- And (DECODE(cr.TYPE, 'CASH', 'STANDARD', 'MISCELLANEOUS') =  :p_receipt_type Or :p_receipt_type Is Null)
		--- And (TRUNC(cr.receipt_date) >= TRUNC(:p_from_date) Or  :p_from_date Is Null)
		--- And (TRUNC(cr.receipt_date) <= TRUNC(:p_to_date)   Or  :p_to_date Is Null)
		--- And (cust.cust_account_id = :p_customer Or :p_customer Is Null)
		--- And (cr.receivables_trx_id = :p_activity_id Or :p_activity_id Is Null)
		--AND (:p_status = UPPER(arpt_sql_func_util.get_lookup_meaning('RECEIPT_CREATION_STATUS', crh_current.status)) or :p_status is null)
		-- START BY MOHAMED.DAGHER 31-05-2016 TO FILTER BY BUSINESS UNTI
		AND APP.CASH_RECEIPT_ID = CR.CASH_RECEIPT_ID
		AND APP.APPLIED_PAYMENT_SCHEDULE_ID = PS_INV.PAYMENT_SCHEDULE_ID(+)
		--     And APP.DISPLAY = 'Y'
		AND CTT.CUST_TRX_TYPE_ID(+) = PS_INV.CUST_TRX_TYPE_ID
		AND APP.CASH_RECEIPT_ID NOT IN
               (
		SELECT
			xlt.source_id_int_1
		FROM
			apps.xla_events xe,
			xla.xla_transaction_entities xlt
		WHERE
			xe.entity_id = xlt.entity_id
			AND xe.application_id = xlt.application_id
			AND xlt.source_application_id = 222
			AND xlt.ledger_id = 2030
			AND xe.process_status_code <> 'P'
			AND xe.event_status_code <> 'P'))
)  temp """


def RME_Collection_ETL():
    spark = fx.spark_app('RME_Collection','4g','4')
    RES = fx.connection(spark,'RES','RMEDB',RME_Collection_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'RME_Collection_Report','overwrite',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('RME_Collection',catchup=False,default_args=default_args,schedule_interval='0 4 * * *',tags=['4'])


RME_CollectionTask= PythonOperator(dag=dag,
                task_id = 'RME_Collection',
                python_callable=RME_Collection_ETL) 


RME_CollectionTask
