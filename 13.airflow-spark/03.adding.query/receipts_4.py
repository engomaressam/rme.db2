import sys
import os
import pendulum
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from airflow.exceptions import AirflowFailException
import mysql.connector as mysql
import oracledb
import pandas as pd

# ✅ Explicitly set the Spark Oracle Client library path
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/oracle/21/client64"
os.environ["ORACLE_HOME"] = "/usr/lib/oracle/21/client64"
os.environ["TNS_ADMIN"] = "/usr/lib/oracle/21/client64/network/admin"

# Import connection settings
import Connections as conn

# Retrieve MySQL credentials from Airflow Variables
mysql_host = Variable.get("MYSQL_HOST")
mysql_user = Variable.get("MYSQL_USER")
mysql_password = Variable.get("MYSQL_PASSWORD")
mysql_database = Variable.get("MYSQL_DATABASE")
mysql_table = "receipts_4"

# Define Oracle ERP Connection Details
oracle_hostname = "10.0.11.59"
oracle_port = 1521
oracle_service_name = "RMEDB"
oracle_username = "RME_DEV"
oracle_password = "PASS21RME"

def erp_to_mysql_etl():
    try:
        print("🔄 Connecting to Oracle ERP with Spark Client...")

        # ✅ Use the Spark-Compatible Oracle Client
        oracledb.init_oracle_client(lib_dir="/usr/lib/oracle/21/client64")

        dsn = oracledb.makedsn(oracle_hostname, oracle_port, service_name=oracle_service_name)
        connection_erp = oracledb.connect(user=oracle_username, password=oracle_password, dsn=dsn)
        cursor_erp = connection_erp.cursor()

        print("✅ Connected to Oracle ERP!")

        # ✅ Query the View Instead of Complex Query
        query_erp = "SELECT * FROM RME_DEV.RME_INV_COLLECTING_STATUS_V"

        print("🔄 Running query on Oracle ERP...")
        cursor_erp.execute(query_erp)
        data = cursor_erp.fetchall()
        columns = [desc[0].strip().upper() for desc in cursor_erp.description]
        df = pd.DataFrame(data, columns=columns)
        print(f"✅ Fetched {len(df)} rows from Oracle ERP.")

        # ✅ Convert dates & handle NULLs
        df = df.where(pd.notna(df), None).replace('', None)

        print("🔌 Connecting to MySQL...")
        mysql_connection = mysql.connect(host=mysql_host, user=mysql_user, password=mysql_password, database=mysql_database)
        mysql_cursor = mysql_connection.cursor()

        print("🗑️ Dropping and recreating table in MySQL...")
        mysql_cursor.execute(f"DROP TABLE IF EXISTS {mysql_table}")
        create_table_query = f"""
        CREATE TABLE {mysql_table} (
            {", ".join([f"{col} VARCHAR(255)" for col in columns])}
        )
        """
        mysql_cursor.execute(create_table_query)
        mysql_connection.commit()

        print("📤 Inserting new data into MySQL table...")
        placeholders = ", ".join(["%s"] * len(df.columns))
        insert_query = f"INSERT INTO {mysql_table} ({', '.join(df.columns)}) VALUES ({placeholders})"
        mysql_cursor.executemany(insert_query, df.values.tolist())
        mysql_connection.commit()

        print(f"✅ Successfully inserted {len(df)} rows into MySQL.")

    except Exception as e:
        print(f"❌ Error during ETL: {e}")
        raise AirflowFailException(e)

    finally:
        if 'cursor_erp' in locals(): cursor_erp.close()
        if 'connection_erp' in locals(): connection_erp.close()
        if 'mysql_cursor' in locals(): mysql_cursor.close()
        if 'mysql_connection' in locals() and mysql_connection.is_connected(): mysql_connection.close()
        print("🔌 Connections closed.")

# Define Airflow DAG
local_tz = pendulum.timezone("Europe/Helsinki")
default_args = {
    'owner': 'gamal',
    'start_date': datetime(2024, 11, 14, tzinfo=local_tz),
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
    'email': ['mohamed.Ghassan@rowad-rme.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}

dag = DAG(
    'receipts_4',
    catchup=False,
    default_args=default_args,
    schedule_interval='30 5 * * *',
    tags=['ERP', 'MySQL', 'Spark']
)

erp_to_mysql_task = PythonOperator(
    dag=dag,
    task_id='erp_to_mysql',
    python_callable=erp_to_mysql_etl
)

erp_to_mysql_task
