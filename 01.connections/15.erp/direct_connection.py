"""
Oracle Database Connection Script
This script attempts to connect directly to the Oracle database using the installed Oracle Instant Client.
"""

import cx_Oracle

# Set the Oracle Client location
cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")

# Connection details 
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"  

# Create a more detailed connection string with additional parameters
# These parameters might help with network connectivity issues
conn_str = f"""
(DESCRIPTION=
    (ADDRESS=
        (PROTOCOL=TCP)
        (HOST={hostname})
        (PORT={port})
    )
    (CONNECT_DATA=
        (SERVICE_NAME={service_name})
    )
    (SECURITY=
        (SSL_SERVER_CERT_DN="")
    )
)
"""

try:
    print(f"Attempting to connect to Oracle database at {hostname}:{port}/{service_name}")
    print(f"Using Oracle Instant Client at C:\\oracle\\instantclient_21_15")
    
    # Try connecting with the detailed connection string
    connection = cx_Oracle.connect(username, password, conn_str)
    print("Connected to Oracle Database!")

    # Test the connection
    cursor = connection.cursor()
    cursor.execute("SELECT 1 FROM DUAL")
    result = cursor.fetchone()
    print(f"Query result: {result}")

    # Close the connection
    cursor.close()
    connection.close()
    print("Connection closed.")
    
except cx_Oracle.Error as error:
    print(f"Error connecting to Oracle: {error}")
    
    # Try an alternative connection method if the first one fails
    try:
        print("\nTrying alternative connection method...")
        dsn_tns = cx_Oracle.makedsn(hostname, port, service_name=service_name)
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
        
        print("Connected to Oracle Database using alternative method!")
        
        # Test the connection
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM DUAL")
        result = cursor.fetchone()
        print(f"Query result: {result}")
        
        # Close the connection
        cursor.close()
        connection.close()
        print("Connection closed.")
        
    except cx_Oracle.Error as error:
        print(f"Error with alternative connection method: {error}")
        print("\nPossible solutions:")
        print("1. Make sure you're on the company network that can access the database server")
        print("2. Check if a VPN connection is required")
        print("3. Verify the database server details (hostname, port, service_name) with your DBA")
        print("4. Ensure the Oracle Instant Client is properly installed and configured")
