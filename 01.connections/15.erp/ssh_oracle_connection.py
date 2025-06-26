import paramiko
import time
import sys

# SSH connection details
ssh_host = "10.10.11.241"
ssh_port = 22
ssh_username = "Rowad\\Omar` Essam2"  # Using PowerShell backtick to escape space
ssh_password = "PMO@1234"  # Your password

# Oracle connection script to run on the remote server
oracle_script = '''
import cx_Oracle

try:
    # Connection details 
    hostname = "10.0.11.59"
    port = 1521
    service_name = "RMEDB"
    username = "RME_DEV"
    password = "PASS21RME"  

    # Set up the connection
    dsn_tns = cx_Oracle.makedsn(hostname, port, service_name=service_name)
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
    print(f"Error: {e}")
'''

def ssh_connect_and_run_oracle():
    try:
        print(f"Connecting to {ssh_host} with username {ssh_username}...")
        
        # Create SSH client
        client = paramiko.SSHClient()
        
        # Set policy to auto-add and accept host keys
        # This addresses the host key verification issue
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        print("Attempting to connect...")
        # Connect to the server with auto-add host key policy
        client.connect(
            hostname=ssh_host, 
            port=ssh_port, 
            username=ssh_username, 
            password=ssh_password, 
            timeout=30,
            allow_agent=False,
            look_for_keys=False,
            banner_timeout=60
        )
        print("SSH connection established successfully!")
        
        # Create a temporary Python script on the remote server
        remote_script_path = "C:\\Temp\\oracle_test.py"
        stdin, stdout, stderr = client.exec_command(f'mkdir -p C:\\Temp')
        
        # Write the Oracle connection script to the remote server
        sftp = client.open_sftp()
        with sftp.file(remote_script_path, 'w') as f:
            f.write(oracle_script)
        sftp.close()
        
        print(f"Oracle connection script created at {remote_script_path}")
        
        # Execute the script on the remote server
        print("Running Oracle connection script...")
        stdin, stdout, stderr = client.exec_command(f'python {remote_script_path}')
        
        # Print output
        print("\nOutput:")
        for line in stdout:
            print(line.strip())
        
        # Print errors if any
        error = stderr.read().decode()
        if error:
            print("\nErrors:")
            print(error)
        
        # Clean up
        stdin, stdout, stderr = client.exec_command(f'del {remote_script_path}')
        
        # Close the connection
        client.close()
        print("SSH connection closed.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    ssh_connect_and_run_oracle()
