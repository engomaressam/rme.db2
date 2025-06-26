"""
Final Oracle Database Connection Attempt
This script includes comprehensive error handling and connection options.
"""

import cx_Oracle
import time
import socket

# Set the Oracle Client location
cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")

# Connection details 
hostname = "10.0.11.59"
port = 1521
service_name = "RMEDB"
username = "RME_DEV"
password = "PASS21RME"

# First, let's check if we can reach the database server
print(f"Testing connectivity to {hostname}:{port}...")
try:
    # Set a shorter timeout for the socket connection test
    socket.setdefaulttimeout(5)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((hostname, port))
    s.close()
    print(f"Successfully connected to {hostname}:{port}")
    network_reachable = True
except Exception as e:
    print(f"Cannot reach {hostname}:{port}: {e}")
    network_reachable = False

if not network_reachable:
    print("\nThe database server is not reachable. Possible reasons:")
    print("1. You're not connected to the company network")
    print("2. There's a firewall blocking the connection")
    print("3. The database server is down or not accepting connections")
    print("4. The hostname or port is incorrect")
    print("\nPossible solutions:")
    print("1. Connect to the company VPN if you're working remotely")
    print("2. Check with your network administrator about firewall rules")
    print("3. Verify the database server details with your DBA")
    print("4. Try connecting through the Windows server at 10.10.11.241 if direct connection is not allowed")
else:
    # Try different connection methods
    connection_methods = [
        {
            "name": "Standard DSN",
            "connect": lambda: cx_Oracle.connect(
                user=username, 
                password=password, 
                dsn=cx_Oracle.makedsn(hostname, port, service_name=service_name)
            )
        },
        {
            "name": "Connection string with DESCRIPTION",
            "connect": lambda: cx_Oracle.connect(
                username, 
                password, 
                f"""(DESCRIPTION=
                    (ADDRESS=
                        (PROTOCOL=TCP)
                        (HOST={hostname})
                        (PORT={port})
                    )
                    (CONNECT_DATA=
                        (SERVICE_NAME={service_name})
                    )
                )"""
            )
        },
        {
            "name": "Easy Connect string",
            "connect": lambda: cx_Oracle.connect(
                username, 
                password, 
                f"{hostname}:{port}/{service_name}"
            )
        },
        {
            "name": "Connection with timeout parameters",
            "connect": lambda: cx_Oracle.connect(
                user=username, 
                password=password, 
                dsn=cx_Oracle.makedsn(hostname, port, service_name=service_name),
                encoding="UTF-8",
                nencoding="UTF-8",
                threaded=True
            )
        }
    ]
    
    success = False
    
    for method in connection_methods:
        try:
            print(f"\nTrying connection method: {method['name']}")
            start_time = time.time()
            connection = method["connect"]()
            end_time = time.time()
            
            print(f"Connected to Oracle Database! (took {end_time - start_time:.2f} seconds)")
            
            # Test the connection
            cursor = connection.cursor()
            cursor.execute("SELECT 1 FROM DUAL")
            result = cursor.fetchone()
            print(f"Query result: {result}")
            
            # Close the connection
            cursor.close()
            connection.close()
            print("Connection closed.")
            
            success = True
            break
            
        except cx_Oracle.Error as error:
            print(f"Error: {error}")
    
    if not success:
        print("\nAll connection methods failed. Recommendations:")
        print("1. Verify that the Oracle Instant Client is correctly installed")
        print("2. Check that you have network access to the database server")
        print("3. Confirm the database credentials and connection details")
        print("4. Consider using a jump server or VPN if direct access is restricted")
        print("5. Consult with your database administrator for assistance")
