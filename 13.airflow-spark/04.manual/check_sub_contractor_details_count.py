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

view_owner = 'RME_DEV'
view_name = 'Sub_contractor_details'

try:
    print("🔄 Connecting to the Oracle ERP database...")
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()
    print("✅ Successfully connected to Oracle ERP!")
    
    query = f"SELECT COUNT(*) FROM {view_owner}.{view_name}"
    cursor.execute(query)
    row_count = cursor.fetchone()[0]
    print(f"✅ Total rows in {view_owner}.{view_name}: {row_count}")
    cursor.close()
    connection.close()
except oracledb.Error as error:
    print(f"❌ Oracle Database error: {error}")
