import oracledb
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

try:
    print("🔄 Connecting to Oracle ERP...")
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor_erp = connection_erp.cursor()
    cursor_erp.execute("SELECT COLUMN_NAME, DATA_TYPE FROM ALL_TAB_COLUMNS WHERE TABLE_NAME = 'RME_PROJECTS_COST_DIST_LINE'")
    columns = cursor_erp.fetchall()
    print("Columns in RME_Projects_Cost_Dist_Line:")
    for col in columns:
        print(col)
    cursor_erp.close()
    connection_erp.close()
except Exception as e:
    print(f"❌ Oracle error: {e}")
