import mysql.connector
from mysql.connector import Error

def check_columns():
    try:
        connection = mysql.connector.connect(
            host='10.10.11.242',
            user='omar2',
            password='Omar_54321',
            database='RME_TEST'
        )
        cursor = connection.cursor()
        
        # Get column names
        cursor.execute("""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'RME_PO_Follow_Up_Report'
        ORDER BY ORDINAL_POSITION
        """)
        
        print("Columns in RME_PO_Follow_Up_Report:")
        for (column_name,) in cursor:
            print(column_name)
            
    except Error as e:
        print(f"Error: {str(e)}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    check_columns()
