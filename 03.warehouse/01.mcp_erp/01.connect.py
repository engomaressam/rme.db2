print("Script started.")
import cx_Oracle

# Connection details 
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"  

dsn_tns = cx_Oracle.makedsn(hostname, port, service_name=service_name)
connection = None  # Initialize connection here

try:
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
    print("Connected to Oracle Database!")

    cursor = connection.cursor()
    cursor.execute("SELECT 1 FROM DUAL")
    result = cursor.fetchone()
    print(result)

except cx_Oracle.Error as error:
    print(f"Error connecting to Oracle: {error}")

finally:
    if connection:
        cursor.close()
        connection.close()
        print("Connection closed.")