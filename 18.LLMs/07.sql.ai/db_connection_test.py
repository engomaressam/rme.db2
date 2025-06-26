import psycopg2
import sys

# PostgreSQL database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'PMO@1234',
    'database': 'postgres'
}

def test_connection():
    """Test the database connection and run a simple query"""
    try:
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database']
        )
        
        print("Connection successful!")
        cursor = conn.cursor()
        
        # Test query on the original table
        print("\nTesting query on po_data.po_followup_merged...")
        try:
            cursor.execute("SELECT COUNT(*) FROM po_data.po_followup_merged")
            count = cursor.fetchone()[0]
            print(f"Query successful! Found {count} rows.")
            
            # Test a more complex query
            cursor.execute("""
                SELECT "VENDOR_NAME", COUNT(*) 
                FROM po_data.po_followup_merged 
                WHERE "PROJECT_NAME" = 'Ring Road El-Qwamiya'
                GROUP BY "VENDOR_NAME"
                LIMIT 5
            """)
            results = cursor.fetchall()
            print("\nSample query results:")
            for row in results:
                print(f"  {row[0]}: {row[1]} records")
                
        except Exception as e:
            print(f"Query failed: {e}")
        
        # Test query on the copied table
        print("\nTesting query on po_data.merged2...")
        try:
            cursor.execute("SELECT COUNT(*) FROM po_data.merged2")
            count = cursor.fetchone()[0]
            print(f"Query successful! Found {count} rows.")
            
            # Test a more complex query
            cursor.execute("""
                SELECT "VENDOR_NAME", COUNT(*) 
                FROM po_data.merged2 
                WHERE "PROJECT_NAME" = 'Ring Road El-Qwamiya'
                GROUP BY "VENDOR_NAME"
                LIMIT 5
            """)
            results = cursor.fetchall()
            print("\nSample query results:")
            for row in results:
                print(f"  {row[0]}: {row[1]} records")
                
        except Exception as e:
            print(f"Query failed: {e}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def fix_rev16():
    """Create a fixed version of rev.16 that ensures it connects to the right database"""
    try:
        with open("16.po.followup.query.ai.rev.16.on.gpu.gemma3.postgres.py", 'r', encoding='utf-8') as source:
            content = source.read()
            
        # Create a fixed version with explicit schema
        with open("16.fixed.py", 'w', encoding='utf-8') as target:
            # Replace the table name to include schema explicitly
            content = content.replace('NEW_TABLE = "po_followup_merged"', 'NEW_TABLE = "po_data.po_followup_merged"')
            content = content.replace("The table name is 'po_followup_merged'", "The table name is 'po_data.po_followup_merged'")
            
            # Update port to avoid conflicts
            content = content.replace("port=7869", "port=7870")
            
            # Write the modified content
            target.write(content)
            
        print("Created fixed version at 16.fixed.py")
        print("Run it with: python 16.fixed.py")
        return True
    except Exception as e:
        print(f"Error creating fixed version: {e}")
        return False

if __name__ == "__main__":
    if test_connection():
        print("\nDatabase connection and queries are working correctly.")
        fix_rev16()
    else:
        print("\nDatabase connection test failed.")
