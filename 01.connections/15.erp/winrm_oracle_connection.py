import subprocess
import sys
import os

# Windows Remote Management connection details
remote_host = "10.10.11.241"
username = "Rowad\\Omar Essam2"
password = "PMO@1234"

def test_winrm_connection():
    print(f"Testing connection to {remote_host} using WinRM...")
    
    # Create a PowerShell command to test WinRM connection
    ps_command = f"""
    $secpasswd = ConvertTo-SecureString '{password}' -AsPlainText -Force
    $cred = New-Object System.Management.Automation.PSCredential ('{username}', $secpasswd)
    $session = New-PSSession -ComputerName {remote_host} -Credential $cred -ErrorAction Stop
    if ($session) {{
        Write-Output "Successfully connected to {remote_host}"
        Remove-PSSession $session
    }}
    """
    
    # Save the PowerShell command to a temporary file
    temp_ps_file = "temp_winrm_test.ps1"
    with open(temp_ps_file, "w") as f:
        f.write(ps_command)
    
    try:
        # Execute the PowerShell script
        result = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_ps_file], 
                               capture_output=True, text=True)
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
            
        if "Successfully connected" in result.stdout:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_ps_file):
            os.remove(temp_ps_file)

def run_oracle_script_via_winrm():
    print(f"Attempting to run Oracle connection script on {remote_host} via WinRM...")
    
    # Oracle connection script
    oracle_script = """
    try {
        # Check if cx_Oracle is installed
        $cxOracle = python -c "import cx_Oracle; print('cx_Oracle is installed')" 2>&1
        if ($cxOracle -match "No module named") {
            Write-Output "cx_Oracle is not installed on the remote server"
            exit
        }
        
        # Create a temporary Python script
        $pythonScript = @"
import cx_Oracle

try:
    # Connection details 
    hostname = '10.0.11.59'
    port = 1521
    service_name = 'RMEDB'
    username = 'RME_DEV'
    password = 'PASS21RME'  

    # Set up the connection
    dsn_tns = cx_Oracle.makedsn(hostname, port, service_name=service_name)
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
    print('Connected to Oracle Database!')

    # Test the connection
    cursor = connection.cursor()
    cursor.execute('SELECT 1 FROM DUAL')
    result = cursor.fetchone()
    print(f'Query result: {result}')

    # Close the connection
    cursor.close()
    connection.close()
    print('Connection closed.')
except Exception as e:
    print(f'Error: {e}')
"@
        
        # Save the script to a temporary file
        $scriptPath = "C:\\Temp\\oracle_test.py"
        New-Item -Path "C:\\Temp" -ItemType Directory -Force | Out-Null
        Set-Content -Path $scriptPath -Value $pythonScript
        
        # Run the script
        Write-Output "Running Oracle connection script..."
        $result = python $scriptPath
        Write-Output $result
        
        # Clean up
        Remove-Item -Path $scriptPath -Force
    }
    catch {
        Write-Output "Error: $_"
    }
    """
    
    # Create a PowerShell command to execute the Oracle script via WinRM
    ps_command = f"""
    $secpasswd = ConvertTo-SecureString '{password}' -AsPlainText -Force
    $cred = New-Object System.Management.Automation.PSCredential ('{username}', $secpasswd)
    
    try {{
        $scriptBlock = [ScriptBlock]::Create(@"
{oracle_script}
"@)
        
        $result = Invoke-Command -ComputerName {remote_host} -Credential $cred -ScriptBlock $scriptBlock -ErrorAction Stop
        Write-Output $result
    }}
    catch {{
        Write-Output "Error: $_"
    }}
    """
    
    # Save the PowerShell command to a temporary file
    temp_ps_file = "temp_winrm_oracle.ps1"
    with open(temp_ps_file, "w") as f:
        f.write(ps_command)
    
    try:
        # Execute the PowerShell script
        result = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_ps_file], 
                               capture_output=True, text=True)
        
        print("Output:")
        print(result.stdout)
        
        if result.stderr:
            print("Errors:")
            print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_ps_file):
            os.remove(temp_ps_file)

if __name__ == "__main__":
    # First test the WinRM connection
    if test_winrm_connection():
        # If connection is successful, run the Oracle script
        run_oracle_script_via_winrm()
    else:
        print("Failed to establish WinRM connection. Cannot run Oracle script.")
