"""
Ollama API Service Starter
This script helps to SSH into a GPU machine and start the Ollama API service.
It provides monitoring and ensures the service stays running.
"""

import paramiko
import time
import socket
import logging
import argparse
import sys
import os
from getpass import getpass

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ollama_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OllamaService")

def check_ssh_connection(hostname, port=22, timeout=5):
    """Check if SSH port is open on the target machine"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((hostname, port))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error checking SSH connection: {str(e)}")
        return False

def start_ollama_service(hostname, username, password=None, key_path=None, port=22):
    """
    SSH into the GPU machine and start the Ollama API service
    
    Args:
        hostname: The hostname or IP of the GPU machine
        username: SSH username
        password: SSH password (optional if using key authentication)
        key_path: Path to SSH private key (optional if using password authentication)
        port: SSH port (default: 22)
    """
    if not check_ssh_connection(hostname, port):
        logger.error(f"SSH port {port} is not open on {hostname}. Please enable SSH on the GPU machine.")
        return False
    
    try:
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the server
        logger.info(f"Connecting to {hostname} as {username}...")
        if key_path and os.path.exists(key_path):
            client.connect(hostname, port=port, username=username, key_filename=key_path)
        else:
            client.connect(hostname, port=port, username=username, password=password)
        
        logger.info("Connected successfully!")
        
        # For Windows: Check if Ollama is installed using PowerShell
        logger.info("Checking if Ollama is installed...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-Command ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source"')
        ollama_path = stdout.read().decode().strip()
        
        if not ollama_path:
            logger.error("Ollama is not installed on the GPU machine.")
            logger.info("Please install Ollama from https://ollama.com/download/windows")
            client.close()
            return False
        
        # Create firewall rule for Ollama API
        create_ollama_firewall_rule(client)
        
        # Check if Ollama service is already running (Windows version)
        logger.info("Checking if Ollama service is already running...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue"')
        ollama_process = stdout.read().decode().strip()
        
        if ollama_process:
            logger.info("Ollama process is already running")
            
            # Check if it's accessible
            stdin, stdout, stderr = client.exec_command('powershell -Command "Invoke-RestMethod -Uri http://localhost:11434/api/tags -Method Get -ErrorAction SilentlyContinue"')
            response = stdout.read().decode().strip()
            
            if "models" in response:
                logger.info("Ollama API is accessible and working properly")
            else:
                logger.warning("Ollama process is running but API might not be accessible")
                logger.info("Restarting Ollama service...")
                client.exec_command('powershell -Command "Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue"')
                time.sleep(2)
                client.exec_command('powershell -Command "Start-Process -FilePath ollama -ArgumentList \'serve\' -WindowStyle Hidden"')
        else:
            # Start Ollama service on Windows
            logger.info("Starting Ollama service...")
            client.exec_command('powershell -Command "Start-Process -FilePath ollama -ArgumentList \'serve\' -WindowStyle Hidden"')
            time.sleep(5)
            
            # Verify service is running
            stdin, stdout, stderr = client.exec_command('powershell -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue"')
            ollama_process = stdout.read().decode().strip()
            
            if ollama_process:
                logger.info("Ollama service started successfully")
            else:
                logger.error("Failed to start Ollama service")
                client.close()
                return False
        
        # Wait for the API to be fully initialized
        logger.info("Waiting for Ollama API to initialize...")
        max_attempts = 10
        for attempt in range(max_attempts):
            stdin, stdout, stderr = client.exec_command('powershell -Command "Invoke-RestMethod -Uri http://localhost:11434/api/tags -Method Get -ErrorAction SilentlyContinue"')
            response = stdout.read().decode().strip()
            
            if "models" in response:
                logger.info("Ollama API is now accessible")
                break
            
            if attempt < max_attempts - 1:
                logger.info(f"Waiting for API to initialize (attempt {attempt + 1}/{max_attempts})...")
                time.sleep(3)
            else:
                logger.warning("Ollama API did not initialize within the expected time")
        
        # Get available models
        logger.info("Checking available models...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Invoke-RestMethod -Uri http://localhost:11434/api/tags -Method Get"')
        models_output = stdout.read().decode()
        
        if "models" in models_output:
            logger.info(f"Available models: {models_output}")
        else:
            logger.warning("Could not retrieve model list. Ollama API might not be fully initialized yet.")
        
        logger.info("Ollama service setup completed successfully")
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error setting up Ollama service: {str(e)}")
        return False

def create_ollama_firewall_rule(client):
    """Create a firewall rule to allow Ollama API traffic"""
    try:
        logger.info("Creating firewall rule for Ollama API (port 11434)...")
        
        # Check if rule already exists
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-NetFirewallRule -Name ollama-api -ErrorAction SilentlyContinue"')
        rule_exists = stdout.read().decode().strip()
        
        if rule_exists:
            logger.info("Firewall rule for Ollama API already exists")
            return True
        
        # Create the rule
        cmd = 'powershell -Command "New-NetFirewallRule -Name ollama-api -DisplayName \"Ollama API\" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow"'
        stdin, stdout, stderr = client.exec_command(cmd)
        error = stderr.read().decode().strip()
        
        if error:
            logger.error(f"Error creating firewall rule: {error}")
            return False
        
        logger.info("Firewall rule created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating firewall rule: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Start Ollama API service on a remote GPU machine")
    parser.add_argument("--host", default="10.10.12.202", help="Hostname or IP of the GPU machine (default: 10.10.12.202)")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--domain", default="ROWAD", help="Domain for domain user (default: ROWAD)")
    parser.add_argument("--user", default="Omar Essam2", help="SSH username (default: Omar Essam2)")
    parser.add_argument("--key", help="Path to SSH private key file")
    
    args = parser.parse_args()
    
    # Format username with domain if provided
    username = f"{args.domain}\\{args.user}" if args.domain else args.user
    
    # Get password if key is not provided
    password = None
    if not args.key:
        password = getpass(f"Enter SSH password for {username}@{args.host}: ")
    
    success = start_ollama_service(
        hostname=args.host,
        username=username,
        password=password,
        key_path=args.key,
        port=args.port
    )
    
    if success:
        logger.info(f"Ollama API should now be accessible at http://{args.host}:11434")
        logger.info("You can now run your chatbot pointing to this API endpoint")
    else:
        logger.error("Failed to set up Ollama service")
        sys.exit(1)

if __name__ == "__main__":
    main()
