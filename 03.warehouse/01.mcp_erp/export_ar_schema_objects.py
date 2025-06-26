import cx_Oracle
import pandas as pd

# Connection details 
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"  

# DSN
dsn_tns = cx_Oracle.makedsn(hostname, port, service_name=service_name)

try:
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
    print("Connected to Oracle Database!")
    cursor = connection.cursor()

    # Query to get all tables and views in AR schema
    query = """
        SELECT 'TABLE' AS OBJECT_TYPE, table_name AS OBJECT_NAME
        FROM all_tables
        WHERE owner = 'AR'
        UNION ALL
        SELECT 'VIEW' AS OBJECT_TYPE, view_name AS OBJECT_NAME
        FROM all_views
        WHERE owner = 'AR'
        ORDER BY OBJECT_TYPE, OBJECT_NAME
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    # Save to Excel
    df = pd.DataFrame(rows, columns=["OBJECT_TYPE", "OBJECT_NAME"])
    output_path = "03.warehouse/01.mcp_erp/AR_schema_objects.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Exported {len(df)} objects to {output_path}")

except cx_Oracle.Error as error:
    print(f"Error: {error}")

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals() and connection:
        connection.close()
        print("Connection closed.") 