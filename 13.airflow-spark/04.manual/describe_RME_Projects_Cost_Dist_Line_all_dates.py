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

# List of possible date columns from previous describe
DATE_COLUMNS = [
    'GL_DATE',
    'TRANSFERRED_DATE',
    'EXPENDITURE_ITEM_DATE',
    'MATURATY_DATE'
]

try:
    print("🔄 Connecting to Oracle ERP...")
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection_erp = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor_erp = connection_erp.cursor()
    for col in DATE_COLUMNS:
        print(f"\nChecking column: {col}")
        cursor_erp.execute(f"SELECT MIN({col}), MAX({col}), COUNT(*) FROM RME_DEV.RME_PROJECTS_COST_DIST_LINE WHERE {col} IS NOT NULL")
        min_date, max_date, count_nonnull = cursor_erp.fetchone()
        print(f"  {col}: min={min_date}, max={max_date}, non-null count={count_nonnull}")
        cursor_erp.execute(f"SELECT {col} FROM RME_DEV.RME_PROJECTS_COST_DIST_LINE WHERE {col} IS NOT NULL AND ROWNUM <= 5")
        print(f"  Sample {col} values:")
        for row in cursor_erp.fetchall():
            print(f"    {row[0]}")
    cursor_erp.close()
    connection_erp.close()
except Exception as e:
    print(f"❌ Oracle error: {e}")
