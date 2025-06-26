import mysql.connector as mysql
import pandas as pd
from mysql.connector import Error

# Database connection details
HOST = "10.10.11.242"
USER = "omar2"
PASSWORD = "Omar_54321"
DATABASE = "RME_TEST"
TABLE_NAME = "RME_Receipts_3"

try:
    # Establish the connection
    cnx = mysql.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )
    
    if cnx.is_connected():
        print("Connection successful!")
        
        # Query to fetch data
        query = f"SELECT * FROM {TABLE_NAME}"
        
        # Load data into Pandas DataFrame
        df = pd.read_sql(query, cnx)
        
        # Save to Excel
        file_name = f"{TABLE_NAME}.xlsx"
        df.to_excel(file_name, index=False)
        print(f"Data saved successfully to {file_name}")
        
except Error as e:
    print(f"Error connecting to database: {e}")

finally:
    # Ensure connection is closed properly
    if 'cnx' in locals() and cnx.is_connected():
        cnx.close()
        print("Connection closed.")
