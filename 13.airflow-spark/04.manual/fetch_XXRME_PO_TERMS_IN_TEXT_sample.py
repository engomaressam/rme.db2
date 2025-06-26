import oracledb
import pandas as pd
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
view_name = 'XXRME_PO_TERMS_IN_TEXT'

try:
    print("🔄 Connecting to the Oracle ERP database...")
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()
    print("✅ Successfully connected to Oracle ERP!")
    
    query = f"SELECT * FROM {view_owner}.{view_name} WHERE ROWNUM <= 100"
    print(f"🔄 Running query: {query}")
    cursor.execute(query)
    columns = [col[0] for col in cursor.description]
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    print(f"✅ Fetched {len(df)} rows.")
    if not df.empty:
        df.to_excel("XXRME_PO_TERMS_IN_TEXT_sample.xlsx", index=False)
        print("✅ Saved sample data to XXRME_PO_TERMS_IN_TEXT_sample.xlsx")
    else:
        print("❌ No data fetched from Oracle.")
    cursor.close()
    connection.close()
except oracledb.Error as error:
    print(f"❌ Oracle Database error: {error}")
