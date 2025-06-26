import mysql.connector
from mysql.connector import Error

def create_simple_table():
    try:
        connection = mysql.connector.connect(
            host='10.10.11.242',
            user='omar2',
            password='Omar_54321',
            database='RME_TEST'
        )
        cursor = connection.cursor()
        
        # Drop table if exists
        cursor.execute("DROP TABLE IF EXISTS po_followup_for_gpt")
        
        # Create a simplified table with just the essential columns
        cursor.execute("""
        CREATE TABLE po_followup_for_gpt (
            id INT AUTO_INCREMENT PRIMARY KEY,
            po_number VARCHAR(50),
            creation_date DATE,
            line_amount DECIMAL(18,2),
            vendor_name VARCHAR(255),
            project_name VARCHAR(255)
        )
        """)
        
        # Copy data from the original table with proper date formatting
        cursor.execute("""
        INSERT INTO po_followup_for_gpt (po_number, creation_date, line_amount, vendor_name, project_name)
        SELECT 
            PO_NUM,
            DATE(POH_CREATION_DATE),
            LINE_AMOUNT,
            VENDOR_NAME,
            PROJECT_NAME
        FROM RME_PO_Follow_Up_Report
        WHERE POH_CREATION_DATE IS NOT NULL
        """)
        
        connection.commit()
        print("Table created and data copied successfully!")
        
        # Show sample data
        cursor.execute("SELECT * FROM po_followup_for_gpt LIMIT 5")
        rows = cursor.fetchall()
        print("\nSample data:")
        for row in rows:
            print(row)
            
    except Error as e:
        print(f"Error: {str(e)}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    create_simple_table()
