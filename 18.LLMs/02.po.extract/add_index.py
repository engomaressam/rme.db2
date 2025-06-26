import mysql.connector
from mysql.connector import Error

def add_index():
    try:
        connection = mysql.connector.connect(
            host='10.10.11.242',
            user='omar2',
            password='Omar_54321',
            database='RME_TEST'
        )
        cursor = connection.cursor()
        
        # Add composite indexes
        cursor.execute("""
        CREATE INDEX idx_vendor_project 
        ON po_followup_for_gpt (vendor_name, project_name)
        """)
        
        cursor.execute("""
        CREATE INDEX idx_date_vendor_project 
        ON po_followup_for_gpt (creation_date, vendor_name, project_name)
        """)
        
        connection.commit()
        print("Index added successfully")
            
    except Error as e:
        print(f"Error: {str(e)}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    add_index()
