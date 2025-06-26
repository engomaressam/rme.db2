from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark.sql.types import *
from datetime import * 
import Connections as conn
import ETLFunctions as fx
from airflow import DAG
from airflow.operators.python_operator import PythonOperator 
import pendulum
from pyspark.sql.functions import *

# ✅ Set the MySQL Connection
load_connection_string = "jdbc:mysql://10.10.11.242:3306/RME_TEST?useUnicode=true&characterEncoding=UTF-8"

# ✅ Use Your New View
Receipts_query = """(
    SELECT * FROM RME_DEV.RME_INV_COLLECTING_STATUS_V
) temp """

# ✅ Define ETL Function
def Receipts_ETL():
    spark = fx.spark_app('receipts_4','2g','2')
    RES = fx.connection(spark,'RES','RMEDB',Receipts_query,'TEMP','ERP')

    # ✅ Convert Dates (Modify as Needed)
    RES = RES.withColumn("RECEIPT_DATE", to_date(col("RECEIPT_DATE"), "yyyy-MM-dd"))
    
    # ✅ Write to MySQL
    fx.WriteFunction(RES, load_connection_string, 'receipts_4_Report', 'overwrite', conn.mysql_username, conn.mysql_password)    

# ✅ Airflow DAG Configuration
local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {
    'owner': 'gamal',
    'start_date': datetime(2024, 11, 14, tzinfo=local_tz),
    "retries": 1,
    "retry_delay": timedelta(minutes=30),
    'email': ['mohamed.Ghassan@rowad-rme.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}

dag = DAG(
    'Receipts',
    catchup=False,
    default_args=default_args,
    schedule_interval='0 8-17 * * *',
    tags=['ERP', 'Spark']
)

Receipts_Task = PythonOperator(
    dag=dag,
    task_id='Receipts',
    python_callable=Receipts_ETL
)

Receipts_Task
