"""
Fix Ollama Connection
This script fixes the Ollama API connection on the GPU machine.
"""

import paramiko
import time
import sys
import os

def print_step(message):
    """Print a step message with formatting"""
    print("\n" + "="*80)
    print(f"STEP: {message}")
    print("="*80)

def main():
    # Connection parameters
    hostname = "10.10.12.202"
    username = "ROWAD\\Omar Essam2"
    password = "PMO@1234"  # Hardcoded password
    
    try:
        print_step("Connecting to GPU machine")
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the server
        print(f"Connecting to {hostname} as {username}...")
        client.connect(hostname, port=22, username=username, password=password)
        
        print("Connected successfully!")
        
        print_step("Stopping any running Ollama process")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue"')
        time.sleep(3)
        
        print_step("Setting environment variables")
        stdin, stdout, stderr = client.exec_command('powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; [Environment]::SetEnvironmentVariable(\'OLLAMA_HOST\', \'0.0.0.0:11434\', \'User\')"')
        print("Set OLLAMA_HOST to 0.0.0.0:11434")
        
        print_step("Starting Ollama service")
        stdin, stdout, stderr = client.exec_command('powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; Start-Process -FilePath ollama -ArgumentList \'serve\' -WindowStyle Hidden"')
        print("Started Ollama service")
        
        # Wait for service to start
        print("Waiting for Ollama service to start...")
        time.sleep(5)
        
        print_step("Creating firewall rule")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-NetFirewallRule -Name ollama-api -ErrorAction SilentlyContinue || New-NetFirewallRule -Name ollama-api -DisplayName \'Ollama API\' -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow"')
        print("Created/verified firewall rule for port 11434")
        
        print_step("Verifying Ollama process")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue"')
        process_output = stdout.read().decode().strip()
        
        if process_output:
            print("Ollama process is running")
        else:
            print("WARNING: Ollama process is not running")
        
        print_step("Testing local API access")
        stdin, stdout, stderr = client.exec_command('powershell -Command "try { $response = Invoke-RestMethod -Uri \'http://localhost:11434/api/tags\' -Method Get -TimeoutSec 3 -ErrorAction Stop; Write-Output \'API is accessible locally\' } catch { Write-Output (\'API error: \' + $_.Exception.Message) }"')
        api_output = stdout.read().decode().strip()
        print(f"Local API test result: {api_output}")
        
        # Close the SSH connection
        client.close()
        print("\nSSH connection closed")
        
        print_step("Connection information")
        print("Ollama API should now be accessible at: http://10.10.12.202:11434")
        print("Update your chatbot to use this URL")
        print("You can test the connection with: python 22.quick.api.check.py")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
