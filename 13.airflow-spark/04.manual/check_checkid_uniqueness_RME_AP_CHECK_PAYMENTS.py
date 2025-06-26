import oracledb
import mysql.connector as mysql
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Oracle ERP connection details
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

# MySQL connection details
db_host = "10.10.11.242"
db_user = "omar2"
db_password = "Omar_54321"
db_name = "RME_TEST"

oracle_view = 'RME_AP_CHECK_PAYMENTS'
mysql_table = 'RME_ap_check_payments_Report'

# Oracle: count rows and distinct CHECK_ID
try:
    print("🔄 Connecting to Oracle ERP...")
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor_erp = connection_erp.cursor()
    cursor_erp.execute(f"SELECT COUNT(*), COUNT(DISTINCT CHECK_ID) FROM RME_DEV.{oracle_view}")
    oracle_total, oracle_distinct = cursor_erp.fetchone()
    print(f"Oracle {oracle_view}: total rows = {oracle_total}, distinct CHECK_ID = {oracle_distinct}")
    cursor_erp.close()
    connection_erp.close()
except Exception as e:
    print(f"❌ Oracle error: {e}")

# MySQL: count rows and distinct CHECK_ID
try:
    print("🔄 Connecting to MySQL...")
    mysql_connection = mysql.connect(host=db_host, user=db_user, password=db_password, database=db_name)
    mysql_cursor = mysql_connection.cursor()
    mysql_cursor.execute(f"SELECT COUNT(*), COUNT(DISTINCT CHECK_ID) FROM {mysql_table}")
    mysql_total, mysql_distinct = mysql_cursor.fetchone()
    print(f"MySQL {mysql_table}: total rows = {mysql_total}, distinct CHECK_ID = {mysql_distinct}")
    mysql_cursor.close()
    mysql_connection.close()
except Exception as e:
    print(f"❌ MySQL error: {e}")
