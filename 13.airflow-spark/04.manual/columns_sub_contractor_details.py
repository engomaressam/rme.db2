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
    
    query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE
        FROM ALL_TAB_COLUMNS
        WHERE OWNER = :1 AND TABLE_NAME = :2
        ORDER BY COLUMN_ID
    """
    cursor.execute(query, [view_owner.upper(), view_name.upper()])
    rows = cursor.fetchall()
    print(f"Columns for {view_owner}.{view_name}:")
    for row in rows:
        col_name, data_type, data_length, data_precision, data_scale = row
        print(f"- {col_name}: {data_type}({data_length})" +
              (f", precision={data_precision}" if data_precision else "") +
              (f", scale={data_scale}" if data_scale else ""))
    cursor.close()
    connection.close()
except oracledb.Error as error:
    print(f"❌ Oracle Database error: {error}")
