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
    # Get min and max GL_DATE and count of non-null
    cursor_erp.execute("SELECT MIN(GL_DATE), MAX(GL_DATE), COUNT(*) FROM RME_DEV.RME_PROJECTS_COST_DIST_LINE WHERE GL_DATE IS NOT NULL")
    min_date, max_date, count_nonnull = cursor_erp.fetchone()
    print(f"GL_DATE min: {min_date}, max: {max_date}, non-null count: {count_nonnull}")
    # Sample a few GL_DATE values
    cursor_erp.execute("SELECT GL_DATE FROM RME_DEV.RME_PROJECTS_COST_DIST_LINE WHERE ROWNUM <= 10")
    print("Sample GL_DATE values:")
    for row in cursor_erp.fetchall():
        print(row[0])
    cursor_erp.close()
    connection_erp.close()
except Exception as e:
    print(f"❌ Oracle error: {e}")
