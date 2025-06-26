from pyspark.sql import SparkSession
from pyspark import SparkConf, conf
from pyspark.sql.types import *
from datetime import * 
import Connections as conn
import ETLFunctions as fx
from airflow import DAG
from airflow.operators.python_operator import PythonOperator 
import pendulum

load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"


omar_query="""(
SELECT
	*
FROM
	apps.HZ_CUST_SITE_USES_ALL
WHERE
	PRIMARY_flag = 'N'
	AND SITE_USE_CODE = 'BILL_TO'
)  temp """


def omar_ETL():
    spark = fx.spark_app('omar','2g','2')
    RES = fx.connection(spark,'RES','RMEDB',omar_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'omar_Report','overwrite',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('omar',catchup=False,default_args=default_args,schedule_interval='5 2 * * *',tags=['2'])


omarTask= PythonOperator(dag=dag,
                task_id = 'omar',
                python_callable=omar_ETL) 


omarTask
