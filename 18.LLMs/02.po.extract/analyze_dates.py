import mysql.connector
from mysql.connector import Error
from collections import defaultdict
import re

def connect_to_database():
    """Connect to MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='10.10.11.242',
            user='omar2',
            password='Omar_54321',
            database='RME_TEST'
        )
        return connection
    except Error as e:
        print(f"Error connecting to database: {e}")
        return None

def analyze_date_columns():
    """Analyze the format of date columns in RME_PO_Follow_Up_Report"""
    connection = connect_to_database()
    if not connection:
        return
    
    cursor = connection.cursor()
    
    try:
        # Get all date-related columns
        date_columns = [
            'POH_CREATION_DATE',
            'POH_CRT_DT_LINE',
            'APPROVED_DATE',
            'APRV_DATE_LINE',
            'NEED_BY_DATE',
            'PROMISED_DATE'
        ]
        
        # For each column, get sample of distinct values
        for column in date_columns:
            print(f"\nAnalyzing {column}:")
            # Get recent values first
            try:
                cursor.execute(f"""
                    SELECT DISTINCT {column}
                    FROM RME_PO_Follow_Up_Report
                    WHERE {column} IS NOT NULL
                    ORDER BY {column} DESC
                    LIMIT 5
                """)
                recent = cursor.fetchall()
            except Error as e:
                print(f"Error getting recent values: {e}")
                recent = []
            
            # Then get oldest values
            try:
                cursor.execute(f"""
                    SELECT DISTINCT {column}
                    FROM RME_PO_Follow_Up_Report
                    WHERE {column} IS NOT NULL
                    ORDER BY {column} ASC
                    LIMIT 5
                """)
                oldest = cursor.fetchall()
            except Error as e:
                print(f"Error getting oldest values: {e}")
                oldest = []
            
            if recent or oldest:
                if recent:
                    print("Most recent values:")
                    for row in recent:
                        print(f"  {row[0]}")
                if oldest:
                    print("\nOldest values:")
                    for row in oldest:
                        print(f"  {row[0]}")
            else:
                print("No non-null values found")
                
            # Get some statistics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT {column}) as distinct_values,
                    COUNT(CASE WHEN {column} IS NULL THEN 1 END) as null_count
                FROM RME_PO_Follow_Up_Report
            """)
            stats = cursor.fetchone()
            print(f"\nStatistics:")
            print(f"  Total rows: {stats[0]}")
            print(f"  Distinct values: {stats[1]}")
            print(f"  Null values: {stats[2]}")
            
            # Get min and max values
            cursor.execute(f"""
                SELECT MIN({column}), MAX({column})
                FROM RME_PO_Follow_Up_Report
                WHERE {column} IS NOT NULL
            """)
            min_max = cursor.fetchone()
            if min_max[0]:
                print(f"  Earliest date: {min_max[0]}")
                print(f"  Latest date: {min_max[1]}")
            
    except Error as e:
        print(f"Error: {e}")
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    analyze_date_columns()
