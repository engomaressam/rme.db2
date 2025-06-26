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


nada_query="""(
		select  *
		from  apps.pa_projects_all  p 
		where  p.ORG_ID = 83 
)  temp """


def nada_ETL():
    spark = fx.spark_app('nada','1g','1')
    RES = fx.connection(spark,'RES','RMEDB',nada_query,'TEMP','ERP')
    fx.WriteFunction(RES ,load_connection_string,'nada_Report','overwrite',conn.mysql_username,conn.mysql_password)    

local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {'owner' : 'gamal',
                'start_date' : datetime(2024,11,14, tzinfo=local_tz),"retries": 1,
                 "retry_delay": timedelta(minutes=30),'email': ['mohamed.Ghassan@rowad-rme.com'],
                  'email_on_failure': True,
                  'email_on_retry': False,}
dag = DAG('nada',catchup=False,default_args=default_args,schedule_interval='30 5 * * *',tags=['5'])


nadaTask= PythonOperator(dag=dag,
                task_id = 'nada',
                python_callable=nada_ETL) 


nadaTask
