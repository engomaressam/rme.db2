"""
Restart Ollama Service on GPU Machine
This script connects to the GPU machine via SSH and restarts the Ollama service
with the correct configuration to accept remote connections.
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
logger = logging.getLogger("RestartOllama")

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
        
        # Stop any running Ollama process
        logger.info("Stopping Ollama service...")
        client.exec_command('powershell -Command "Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue"')
        time.sleep(2)
        
        # Configure Ollama to listen on all interfaces
        logger.info("Configuring Ollama to listen on all interfaces...")
        client.exec_command('powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; [Environment]::SetEnvironmentVariable(\'OLLAMA_HOST\', \'0.0.0.0:11434\', \'User\')"')
        
        # Start Ollama with the correct configuration
        logger.info("Starting Ollama service...")
        client.exec_command('powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; Start-Process -FilePath ollama -ArgumentList \'serve\' -WindowStyle Hidden"')
        
        # Wait for service to start
        logger.info("Waiting for Ollama service to start...")
        time.sleep(5)
        
        # Create firewall rule if it doesn't exist
        logger.info("Creating firewall rule for Ollama API...")
        client.exec_command('powershell -Command "Get-NetFirewallRule -Name ollama-api -ErrorAction SilentlyContinue || New-NetFirewallRule -Name ollama-api -DisplayName \'Ollama API\' -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow"')
        
        logger.info("Ollama service has been restarted and configured to accept remote connections")
        logger.info("You can now access the Ollama API at http://10.10.12.202:11434")
        
        # Close the SSH connection
        client.close()
        logger.info("SSH connection closed")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
