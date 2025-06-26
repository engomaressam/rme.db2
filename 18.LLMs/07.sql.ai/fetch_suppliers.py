import psycopg2
import sys

# Set console encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')

# Connect to the database
try:
    conn = psycopg2.connect(
        host='localhost',
        user='postgres',
        password='PMO@1234',
        database='postgres'
    )
    
    cursor = conn.cursor()
    
    # Execute query to get unique supplier names
    cursor.execute('SELECT DISTINCT "VENDOR_NAME" FROM public.po_followup_rev19 WHERE "VENDOR_NAME" IS NOT NULL ORDER BY "VENDOR_NAME"')
    
    # Fetch all results
    results = cursor.fetchall()
    
    # Print the results in a format that can be copied into the code
    print('UNIQUE_SUPPLIERS = [')
    for row in results:
        if row[0]:
            try:
                # Try to safely print the supplier name
                safe_name = row[0].encode('ascii', 'backslashreplace').decode('ascii')
                print(f'    "{safe_name}",')    
            except Exception as e:
                print(f'    # Could not encode: {str(e)}')
    print(']')
    
    # Also save to a file
    with open('suppliers_list.txt', 'w', encoding='utf-8') as f:
        f.write('UNIQUE_SUPPLIERS = [\n')
        for row in results:
            if row[0]:
                f.write(f'    "{row[0]}",\n')
        f.write(']\n')
    
    # Close the connection
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {str(e)}")
