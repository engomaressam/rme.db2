import oracledb

# Explicitly set the Oracle Client location
oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")

# Connection details 
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"  

# Connection string
dsn = f"{hostname}:{port}/{service_name}"

try:
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connected to Oracle Database!")

    cursor = connection.cursor()
    cursor.execute("SELECT 1 FROM DUAL")
    result = cursor.fetchone()
    print(result)

except oracledb.Error as error:
    print(f"Error connecting to Oracle: {error}")

finally:
    if 'connection' in locals() and connection:
        connection.close()
        print("Connection closed.")
