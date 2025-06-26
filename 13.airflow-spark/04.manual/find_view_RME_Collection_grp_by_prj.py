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

def find_view(view_name, schemas):
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()
    found = False
    for schema in schemas:
        print(f"Searching for view '{view_name}' in schema '{schema}'...")
        cursor.execute(f"SELECT VIEW_NAME FROM ALL_VIEWS WHERE OWNER = :1 AND VIEW_NAME = :2", [schema.upper(), view_name.upper()])
        result = cursor.fetchall()
        if result:
            print(f"✅ Found view '{view_name}' in schema '{schema}'.")
            found = True
        else:
            print(f"❌ View '{view_name}' not found in schema '{schema}'.")
    cursor.close()
    connection.close()
    if not found:
        print(f"View '{view_name}' was not found in any of the specified schemas.")

if __name__ == "__main__":
    find_view('RME_Collection_grp_by_prj', ['RME_DEV', 'APPS'])
