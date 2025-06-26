import cx_Oracle
import pandas as pd
import os

# Connection details 
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"  

# DSN
dsn_tns = cx_Oracle.makedsn(hostname, port, service_name=service_name)

# Output directory
output_dir = "03.warehouse/01.mcp_erp/schema_metadata"
os.makedirs(output_dir, exist_ok=True)

schemas = ['AP', 'AR']

try:
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
    print("Connected to Oracle Database!")
    cursor = connection.cursor()

    for schema in schemas:
        print(f"Processing schema: {schema}")
        # Query for tables metadata
        table_query = f"""
            SELECT
                'TABLE' AS OBJECT_TYPE,
                t.table_name AS OBJECT_NAME,
                t.num_rows,
                t.last_analyzed,
                c.column_name,
                c.data_type,
                c.data_length,
                c.nullable,
                c.data_default
            FROM all_tables t
            LEFT JOIN all_tab_columns c
                ON t.owner = c.owner AND t.table_name = c.table_name
            WHERE t.owner = '{schema}'
        """
        # Query for views metadata
        view_query = f"""
            SELECT
                'VIEW' AS OBJECT_TYPE,
                v.view_name AS OBJECT_NAME,
                NULL AS num_rows,
                NULL AS last_analyzed,
                c.column_name,
                c.data_type,
                c.data_length,
                c.nullable,
                c.data_default
            FROM all_views v
            LEFT JOIN all_tab_columns c
                ON v.owner = c.owner AND v.view_name = c.table_name
            WHERE v.owner = '{schema}'
        """
        # Combine and fetch
        full_query = f"{table_query} UNION ALL {view_query} ORDER BY OBJECT_TYPE, OBJECT_NAME, column_name"
        cursor.execute(full_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        # Save to Excel
        output_path = os.path.join(output_dir, f"{schema}_metadata.xlsx")
        df.to_excel(output_path, index=False)
        print(f"Exported {len(df)} rows for schema {schema} to {output_path}")

except cx_Oracle.Error as error:
    print(f"Error: {error}")

finally:
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals() and connection:
        connection.close()
        print("Connection closed.") 