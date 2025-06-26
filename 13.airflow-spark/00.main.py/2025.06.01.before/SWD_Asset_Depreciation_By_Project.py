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


SWD_Asset_Depreciation_By_Project_query="""(
   SELECT
	BOOK_TYPE_CODE,
	to_char(DATE_PLACED_IN_SERVICE,'YYYY-MM-DD') DATE_PLACED_IN_SERVICE,
	ASSET_ID,
	ASSET_NUMBER,
	DESCRIPTION,
	MAJOR_CATEGORY,
	MINOR_CATEGORY,
	EXPEN_TYPE,
	task,
	PROJECTS,
	PRJ_PROJECT,
	DEPT,
	LOCATIONS,
	CURRENT_UNITS,
	UNITS_ASSIGNED,
	DEPRN_PER_UNIT,
	DEPRN_AMOUNT,
	ADJUSTED_COST,
	PERIOD_NAME,
	EXPENSE_ACCOUNT ,
	PERIOD_counter
FROM
	apps.rme_fa_history
where 
     to_char(DATE_PLACED_IN_SERVICE,'YYYY-MM-DD') BETWEEN TO_CHAR(TRUNC(SYSDATE, 'YEAR'), 'YYYY-MM-DD') AND 
      TO_CHAR(TRUNC(SYSDATE) - 1, 'YYYY-MM-DD') 
)  temp """


def delete_data():
        db = mysql.connect(
         host = "10.10.11.242",
        user = conn.mysql_username,
        passwd = conn.mysql_password,database = "RME_TEST"
    )
        cursor = db.cursor()
        operation = "DELETE FROM RME_TEST.SWD_Asset_Depreciation_By_Project_Report WHERE DATE_PLACED_IN_SERVICE BETWEEN DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL DAYOFYEAR(CURDATE()) - 1 DAY), '%Y-%m-%d') AND DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 DAY), '%Y-%m-%d');"
        cursor.execute(operation)
        db.commit()

def SWD_Asset_Depreciation_By_Project_ETL():
    spark = fx.spark_app('SWD_Asset_Depreciation_By_Project','4g','4')
    RES = fx.connection(spark,'RES','RMEDB',SWD_Asset_Depreciation_By_Project_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'SWD_Asset_Depreciation_By_Project_Report','append',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('SWD_Asset_Depreciation_By_Project',catchup=False,default_args=default_args,schedule_interval='00 6 * * *',tags=['3'])


delete_dataTask= PythonOperator(dag=dag,
                task_id = 'delete_dataTask',
                python_callable=delete_data) 


SWD_Asset_Depreciation_By_ProjectTask= PythonOperator(dag=dag,
                task_id = 'SWD_Asset_Depreciation_By_Project',
                python_callable=SWD_Asset_Depreciation_By_Project_ETL) 


delete_dataTask >> SWD_Asset_Depreciation_By_ProjectTask
