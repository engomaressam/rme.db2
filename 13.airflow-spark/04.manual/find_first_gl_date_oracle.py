import oracledb
from datetime import datetime

# Oracle ERP connection details
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

def get_first_gl_date():
    try:
        dsn = oracledb.makedsn(hostname, port, service_name=service_name)
        conn = oracledb.connect(user=username, password=password, dsn=dsn)
        cursor = conn.cursor()
        # Use the same WHERE clause as your ETL but MIN(gl_date)
        query = '''
        SELECT MIN(dis_ln.gl_date)
        FROM apps.pa_cost_distribution_lines_all dis_ln
        WHERE dis_ln.amount != 0
        '''
        cursor.execute(query)
        result = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        if result:
            print(f"First GL_DATE with data: {result.strftime('%Y-%m-%d')}")
        else:
            print("No data found in Oracle.")
    except Exception as e:
        print(f"❌ Oracle error: {e}")

if __name__ == "__main__":
    get_first_gl_date()
