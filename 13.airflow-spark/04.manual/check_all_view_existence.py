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

# List of views to check
views_to_check = [
    'RME_INV_COLLECTING_STATUS_V',
    'RME_Collection_grp_by_prj',
    'Sub_contractor_details',
    'RME_Suppliers_Balance_dist',
    'XXRME_PO_TERMS_IN_TEXT'
]

schemas_to_check = ['RME_DEV', 'APPS']

def check_view_exists(view_name, schemas):
    dsn = oracledb.makedsn(hostname, port, service_name=service_name)
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()
    found = False
    for schema in schemas:
        cursor.execute("SELECT VIEW_NAME FROM ALL_VIEWS WHERE OWNER = :1 AND VIEW_NAME = :2", [schema.upper(), view_name.upper()])
        result = cursor.fetchall()
        if result:
            print(f"✅ View '{view_name}' found in schema '{schema}'.")
            found = True
    if not found:
        print(f"❌ View '{view_name}' NOT found in any checked schema.")
    cursor.close()
    connection.close()

if __name__ == "__main__":
    print("Checking existence of views in schemas: RME_DEV, APPS\n")
    for view in views_to_check:
        check_view_exists(view, schemas_to_check)
