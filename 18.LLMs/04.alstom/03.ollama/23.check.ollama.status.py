"""
Check Ollama Status on GPU Machine
This script connects to the GPU machine via SSH and checks the status of the Ollama service.
"""

import paramiko
import time
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OllamaStatus")

def main():
    # Connection parameters
    hostname = "10.10.12.202"
    username = "ROWAD\\Omar Essam2"
    password = "PMO@1234"  # Hardcoded password
    
    try:
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the server
        logger.info(f"Connecting to {hostname} as {username}...")
        client.connect(hostname, port=22, username=username, password=password)
        
        logger.info("Connected successfully!")
        
        # Check if Ollama process is running
        logger.info("Checking if Ollama process is running...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue | Select-Object Id, CPU, WS, PM, Name"')
        process_output = stdout.read().decode().strip()
        
        if process_output:
            logger.info(f"Ollama process is running:\n{process_output}")
        else:
            logger.warning("Ollama process is not running")
        
        # Check environment variables
        logger.info("Checking Ollama environment variables...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "[Environment]::GetEnvironmentVariable(\'OLLAMA_HOST\', \'User\')"')
        host_env = stdout.read().decode().strip()
        
        if host_env:
            logger.info(f"OLLAMA_HOST environment variable is set to: {host_env}")
        else:
            logger.warning("OLLAMA_HOST environment variable is not set")
        
        # Check if port 11434 is listening
        logger.info("Checking if port 11434 is listening...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-NetTCPConnection -LocalPort 11434 -ErrorAction SilentlyContinue | Select-Object LocalAddress, LocalPort, State"')
        port_output = stdout.read().decode().strip()
        
        if port_output:
            logger.info(f"Port 11434 is listening:\n{port_output}")
        else:
            logger.warning("Port 11434 is not listening")
        
        # Check firewall rule
        logger.info("Checking firewall rule for Ollama API...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-NetFirewallRule -Name ollama-api -ErrorAction SilentlyContinue | Select-Object Name, DisplayName, Enabled, Direction, Action"')
        firewall_output = stdout.read().decode().strip()
        
        if firewall_output:
            logger.info(f"Firewall rule for Ollama API exists:\n{firewall_output}")
        else:
            logger.warning("Firewall rule for Ollama API does not exist")
        
        # Try to access the API locally on the GPU machine
        logger.info("Testing Ollama API locally on the GPU machine...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "try { $response = Invoke-RestMethod -Uri \'http://localhost:11434/api/tags\' -Method Get -TimeoutSec 3 -ErrorAction Stop; $response | ConvertTo-Json -Depth 3 } catch { $_.Exception.Message }"')
        api_output = stdout.read().decode().strip()
        
        if "models" in api_output:
            logger.info(f"Ollama API is accessible locally on the GPU machine:\n{api_output[:500]}...")
        else:
            logger.warning(f"Ollama API is not accessible locally on the GPU machine: {api_output}")
        
        # Close the SSH connection
        client.close()
        logger.info("SSH connection closed")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
