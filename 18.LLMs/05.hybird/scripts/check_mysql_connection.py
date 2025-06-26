"""
MySQL Connection Test Script

This script tests the MySQL connection using environment variables
and counts the number of records in the po.pdfs table.
"""
import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MySQL Configuration
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', '10.10.11.242'),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'user': os.getenv('MYSQL_USER', 'omar2'),
    'password': os.getenv('MYSQL_PASSWORD', 'Omar_54321'),
    'database': os.getenv('MYSQL_DATABASE', 'RME_TEST'),
    'raise_on_warnings': True
}

def test_mysql_connection():
    """Test MySQL connection and count records in po.pdfs"""
    try:
        # Create connection using config
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        
        if conn.is_connected():
            cursor = conn.cursor()
            
            # Count records in po.pdfs
            cursor.execute("SELECT COUNT(*) FROM `po.pdfs`")
            count = cursor.fetchone()[0]
            print(f"[SUCCESS] Connected to MySQL database: {MYSQL_CONFIG['database']}")
            print(f"[INFO] Total records in po.pdfs table: {count}")
            
            # Get table structure for verification
            cursor.execute("DESCRIBE `po.pdfs`")
            columns = cursor.fetchall()
            print("\nTable structure (po.pdfs):")
            for col in columns:
                print(f"- {col[0]}: {col[1]}")
            
            cursor.close()
            conn.close()
            return True
            
    except Error as e:
        print(f"[ERROR] Connecting to MySQL: {e}")
        return False
    
    return False

if __name__ == "__main__":
    print("Testing MySQL connection and checking po.pdfs table...")
    print(f"Connection details: mysql://{MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}")
    
    success = test_mysql_connection()
    
    if not success:
        print("\n[ERROR] Connection failed. Please check:")
        print("1. MySQL server is running")
        print("2. Host, username, and password are correct")
        print("3. User has proper permissions")
        print("4. Table 'po.pdfs' exists in the database")
        print("\nCurrent configuration:", MYSQL_CONFIG)
