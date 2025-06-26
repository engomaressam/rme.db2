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


Vendors_List_query="""(
   SELECT
	DISTINCT
	----------AYA------
	-- IPY.BANK_ACCOUNT_NUMBER BANCK_ACC_NO,     ------AYA REDA-----11-10-2022
	a.VENDOR_ID,
	a.SEGMENT1 VENDOR_NUM,
	a.VENDOR_NAME,
	pvc.FIRST_NAME contact_name,
	---   a.vat_registration_num,
	-----NVL (a.Individual_1099, a.Num_1099) Taxpayer_Id,
       (
	SELECT
		DISTINCT IPY.BANK_ACCOUNT_NUMBER
	FROM
		apps.iby_external_bank_accounts_v IPY
	WHERE
		IPY.PRIMARY_ACCT_OWNER_PARTY_ID = hp.PARTY_ID
		AND a.PARTY_ID = IPY.PRIMARY_ACCT_OWNER_PARTY_ID
		AND ROWNUM = 1)
          BANCK_ACC_NO,
	(
	SELECT
		Concatenated_Segments
	FROM
		apps.Gl_Code_Combinations_Kfv Gcck
	WHERE
		Code_Combination_Id = APSS.Accts_Pay_Code_Combination_Id)
          LIAB_ACC,
	(
	SELECT
		Concatenated_Segments
	FROM
		apps.Gl_Code_Combinations_Kfv Gcck
	WHERE
		Code_Combination_Id = APSS.Prepay_Code_Combination_Id)
          PREPAYMENT_ACCOUNT,
	(
	SELECT
		Concatenated_Segments
	FROM
		Apps.Gl_Code_Combinations_Kfv Gcck
	WHERE
		Code_Combination_Id = APSS.Future_Dated_Payment_Ccid)
          Future_Acc,
	(
	SELECT
		a.TERRITORY_SHORT_NAME
	FROM
		apps.FND_TERRITORIES_VL a
	WHERE
		A.TERRITORY_CODE = hp.COUNTRY)
          COUNTRY,
	APSS.Vendor_Site_Code SITE_NAME,
	hp.PARTY_NUMBER,
	--hp.EMAIL_ADDRESS,
	pvc.EMAIL_ADDRESS,
	pvc.PHONE,
	pvc.TITLE,
	hp.PERSON_PRE_NAME_ADJUNCT,
	pvc.URL
	-- IPY.currency_code
FROM
	apps.ap_suppliers a,
	apps.ap_supplier_sites_all apss,
	apps.PO_VENDOR_CONTACTS pvc,
	apps.hz_parties hp
	--
 WHERE     1 = 1
	--and a.vendor_id = b.vendor_id
	AND pvc.VENDOR_ID(+) = a.VENDOR_ID
	AND a.VENDOR_ID = apss.VENDOR_ID
	-- AND pvs.ACCTS_PAY_CODE_COMBINATION_ID = gcc.CODE_COMBINATION_ID(+)
	-- AND a.VENDOR_ID = 101088 --4333---70974
	--AND PO.PO_HEADER_ID = B.PO_HEADER_ID
	AND a.PARTY_ID = hp.PARTY_ID
)  temp """


def Vendors_List_ETL():
    spark = fx.spark_app('RME_Vendors_List','4g','4')
    RES = fx.connection(spark,'RES','RMEDB',Vendors_List_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'RME_Vendors_List_Report','overwrite',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('Vendors_List',catchup=False,default_args=default_args,schedule_interval='30 5 * * *',tags=['4'])


Vendors_ListTask= PythonOperator(dag=dag,
                task_id = 'Vendors_List',
                python_callable=Vendors_List_ETL) 


Vendors_ListTask
