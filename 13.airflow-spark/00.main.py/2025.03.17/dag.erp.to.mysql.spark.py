import sys
import pendulum
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models import Variable
from airflow.exceptions import AirflowFailException
from pyspark.sql import SparkSession

# Import connection settings
import Connections as conn

# Retrieve MySQL credentials from Airflow Variables
mysql_host = Variable.get("MYSQL_HOST")
mysql_user = Variable.get("MYSQL_USER")
mysql_password = Variable.get("MYSQL_PASSWORD")
mysql_database = Variable.get("MYSQL_DATABASE")
mysql_table = "RME_Receipts_3"

# Define Oracle ERP Connection Details
oracle_hostname = "10.0.11.59"
oracle_port = 1521
oracle_service_name = "RMEDB"
oracle_username = "APPS"
oracle_password = "your_oracle_password"


def erp_to_mysql_spark():
    try:
        print("🚀 Initializing Spark Session...")
        spark = SparkSession.builder \
            .appName("ERP_to_MySQL") \
            .config("spark.jars", "/path/to/mysql-connector-java.jar,/path/to/ojdbc8.jar") \
            .getOrCreate()
        
        print("🔄 Connecting to Oracle ERP via Spark...")
        jdbc_url = f"jdbc:oracle:thin:@{oracle_hostname}:{oracle_port}/{oracle_service_name}"
        oracle_properties = {
            "user": oracle_username,
            "password": oracle_password,
            "driver": "oracle.jdbc.OracleDriver"
        }

        # Define Query
        query_erp = """
        SELECT DISTINCT
                ACR.cash_receipt_id receipt_id,
                ACR.receipt_number,
                trx_head.org_id,
                ppa.name receipt_prj_name,
                acr.attribute1 receipt_prj_code,
                ACR.amount receipt_amount,
                acr.RECEIPT_DATE,
                trx_head.CUSTOMER_TRX_ID,
                TRX_HEAD.TRX_NUMBER Inv_num,
                ARA.AMOUNT_APPLIED,
                ARA.attribute1,
                ( select nvl(sum(AMOUNT),0)
        FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
        where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
        and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
        and adj.status NOT IN ('R', 'U')
        and RT.NAME in ('With Holding Tax')) With_Holding_Tax,
        
        ( select nvl(sum(AMOUNT),0) 
        FROM AR_ADJUSTMENTS_ALL adj, AR_RECEIVABLES_TRX_all RT
        where TRX_HEAD.CUSTOMER_TRX_ID = adj.CUSTOMER_TRX_ID
        and ADJ.RECEIVABLES_TRX_ID = RT.RECEIVABLES_TRX_ID
        and adj.status NOT IN ('R', 'U')
        and RT.NAME in ( 'Stamps','Stamp Tax')) Stamp,

        -- (Retention, Social, Tax, Other Conditions Logic remains unchanged)
        
                TRX_HEAD.INVOICE_CURRENCY_CODE CURRENCY,
                (SELECT SUM (
                            NVL (tl.QUANTITY_INVOICED, 0)
                            * NVL (tl.UNIT_SELLING_PRICE, 1))
                    FROM RA_CUSTOMER_TRX_LINES_ALL TL
                WHERE tl.CUSTOMER_TRX_ID = TRX_HEAD.CUSTOMER_TRX_ID)
                    Transaction_Amount,
                (SELECT NVL (SUM (extended_amount), 0)
                    FROM ra_customer_trx_all trx_head,
                        ra_customer_trx_lines_all trx_line
                WHERE     trx_head.customer_trx_id = trx_line.customer_trx_id(+)
                        AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
                        AND trx_line.line_type IN ('TAX'))
                    Tax_Amount,

                (SELECT aps.AMOUNT_DUE_REMAINING
                    FROM ar_payment_schedules_all aps,
                        ra_customer_trx_all trx_head,
                        ra_customer_trx_lines_all trx_line
                WHERE     trx_head.customer_trx_id = trx_line.customer_trx_id(+)
                        AND TRX_HEAD.CUSTOMER_TRX_ID(+) = ara.APPLIED_CUSTOMER_TRX_ID
                        AND TRX_HEAD.customer_trx_id = aps.customer_trx_id(+)
                        AND TRX_HEAD.org_id = aps.org_id
                        AND ara.APPLIED_PAYMENT_SCHEDULE_ID = aps.PAYMENT_SCHEDULE_ID
                        AND ROWNUM = 1)
                    AMOUNT_DUE_REMAINING,

                hcaa.ACCOUNT_NUMBER Customer_No,
                hcaa.ACCOUNT_NAME Customer_Name

        FROM AR_CASH_RECEIPTS_ALL acr
        LEFT JOIN pa_projects_all ppa ON ppa.segment1 = acr.attribute1
        LEFT JOIN AR_RECEIVABLE_APPLICATIONS_ALL ara ON acr.cash_receipt_id = ara.cash_receipt_id
        LEFT JOIN RA_CUSTOMER_TRX_ALL trx_head ON trx_head.CUSTOMER_TRX_ID = ara.APPLIED_CUSTOMER_TRX_ID
        LEFT JOIN HZ_CUST_ACCOUNTS hcaa ON hcaa.CUST_ACCOUNT_ID = trx_head.BILL_TO_CUSTOMER_ID
        LEFT JOIN hr_operating_units hro ON acr.org_id = hro.organization_id 

        WHERE acr.org_id = 83  -- ✅ Hardcoded organization ID
        AND ara.reversal_gl_date IS NULL
        AND ara.status IN ('ACC', 'APP')

        GROUP BY ACR.cash_receipt_id,
                ACR.receipt_number,
                trx_head.org_id,
                ppa.name,
                acr.attribute1,
                ACR.amount,
                acr.RECEIPT_DATE,
                ara.apply_date,
                trx_head.CUSTOMER_TRX_ID,
                ara.APPLIED_CUSTOMER_TRX_ID,
                ara.cash_receipt_id,
                TRX_HEAD.INTERFACE_HEADER_ATTRIBUTE1,
                ARA.AMOUNT_APPLIED,
                ara.attribute1,
                ara.status,
                TRX_HEAD.EXCHANGE_RATE,
                TRX_HEAD.CUST_TRX_TYPE_ID,
                TRX_HEAD.TRX_NUMBER,
                TRX_HEAD.TRX_DATE,
                TRX_HEAD.INVOICE_CURRENCY_CODE,
                TRX_HEAD.ATTRIBUTE2,
                hcaa.ACCOUNT_NUMBER,
                hcaa.ACCOUNT_NAME,
                ara.APPLIED_PAYMENT_SCHEDULE_ID,
                TRX_HEAD.TERM_ID
        """

        df = spark.read.jdbc(url=jdbc_url, table=query_erp, properties=oracle_properties)
        print(f"✅ Fetched {df.count()} rows from Oracle ERP.")

        # Convert date columns
        df = df.withColumn("RECEIPT_DATE", df["RECEIPT_DATE"].cast("date"))

        print("🔌 Connecting to MySQL via Spark...")
        mysql_url = f"jdbc:mysql://{mysql_host}/{mysql_database}?user={mysql_user}&password={mysql_password}"
        
        df.write \
            .format("jdbc") \
            .option("url", mysql_url) \
            .option("dbtable", mysql_table) \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .mode("overwrite") \
            .save()

        print(f"✅ Successfully inserted {df.count()} rows into MySQL using Spark.")

    except Exception as e:
        print(f"❌ Error during ETL: {e}")
        raise AirflowFailException(e)

    finally:
        spark.stop()
        print("🔌 Spark Session Stopped.")


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
    'erp_to_mysql_spark_etl',
    catchup=False,
    default_args=default_args,
    schedule_interval='30 5 * * *',
    tags=['ERP', 'MySQL', 'Spark']
)

erp_to_mysql_spark_task = PythonOperator(
    dag=dag,
    task_id='erp_to_mysql_spark',
    python_callable=erp_to_mysql_spark
)

erp_to_mysql_spark_task
