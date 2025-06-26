import mysql.connector
from mysql.connector import Error

def connect_to_database():
    """Connect to MySQL database"""
    print("Connecting to database...")
    try:
        connection = mysql.connector.connect(
            host='10.10.11.242',
            user='omar2',
            password='Omar_54321',
            database='RME_TEST'
        )
        print("Connected successfully")
        return connection
    except Error as e:
        print(f"Error connecting to database: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_sample_rows():
    """Get a few complete rows to understand the data format"""
    connection = connect_to_database()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        print("Executing query...")
        cursor.execute("""
            SELECT POH_CREATION_DATE, POH_CRT_DT_LINE, APPROVED_DATE, NEED_BY_DATE
            FROM RME_PO_Follow_Up_Report
            WHERE POH_CREATION_DATE IS NOT NULL
            ORDER BY POH_CREATION_DATE DESC
            LIMIT 3
        """)
        
        print("Getting results...")
        rows = cursor.fetchall()
        
        if not rows:
            print("No rows found!")
            return
            
        print(f"\nFound {len(rows)} rows:")
        for row in rows:
            print(f"\nDates in row:")
            print(f"POH_CREATION_DATE: {row[0]}")
            print(f"POH_CRT_DT_LINE: {row[1]}")
            print(f"APPROVED_DATE: {row[2]}")
            print(f"NEED_BY_DATE: {row[3]}")
            print("-" * 50)
            
    except Error as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    check_sample_rows()
