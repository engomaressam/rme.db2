import mysql.connector
from mysql.connector import Error

def check_table():
    try:
        connection = mysql.connector.connect(
            host='10.10.11.242',
            user='omar2',
            password='Omar_54321',
            database='RME_TEST'
        )
        cursor = connection.cursor()
        
        # Check row count
        cursor.execute("SELECT COUNT(*) FROM po_followup_for_gpt")
        count = cursor.fetchone()[0]
        print(f"Total rows: {count}")
        
        # Check sample data
        cursor.execute("""
        SELECT po_number, creation_date, line_amount, vendor_name 
        FROM po_followup_for_gpt 
        ORDER BY creation_date DESC 
        LIMIT 5
        """)
        
        print("\nRecent POs:")
        print("PO Number | Creation Date | Amount | Vendor")
        print("-" * 60)
        for row in cursor:
            print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
            
    except Error as e:
        print(f"Error: {str(e)}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

if __name__ == "__main__":
    check_table()
