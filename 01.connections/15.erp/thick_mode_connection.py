"""
Oracle Database Connection using Thick Mode
This script attempts to connect to an Oracle database using thick mode,
which might work better in complex network environments.
"""

try:
    # First try to import oracledb (newer library)
    import oracledb
    print("Using oracledb library")
    
    # Initialize the Oracle Client in thick mode
    oracledb.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")
    
    # Connection details
    hostname = "10.0.11.59"
    port = 1521
    service_name = "RMEDB"
    username = "RME_DEV"
    password = "PASS21RME"
    
    # Create the connection string
    dsn = f"{hostname}:{port}/{service_name}"
    
    print(f"Attempting to connect to {dsn} as {username}...")
    
    # For thick mode connections in oracledb, we don't need a special parameter
    # The init_oracle_client call already enables thick mode
    connection = oracledb.connect(user=username, password=password, dsn=dsn)
    
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
    
except ImportError:
    # Fall back to cx_Oracle if oracledb is not available
    print("oracledb not found, trying cx_Oracle...")
    
    try:
        import cx_Oracle
        
        # Initialize the Oracle Client
        cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient_21_15")
        
        # Connection details
        hostname = "10.0.11.59"
        port = 1521
        service_name = "RMEDB"
        username = "RME_DEV"
        password = "PASS21RME"
            
        # Create the connection string
        dsn_tns = cx_Oracle.makedsn(hostname, port, service_name=service_name)
            
        print(f"Attempting to connect to {dsn_tns} as {username}...")
            
        # Connect to the database
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
            
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
        
    except Exception as e:
        print(f"Error with cx_Oracle: {e}")
        
except Exception as e:
    print(f"Error with oracledb: {e}")
