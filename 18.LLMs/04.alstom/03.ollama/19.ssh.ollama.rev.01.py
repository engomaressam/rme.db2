"""
GPU Machine Ollama Checker and Starter
This script connects to the GPU machine via SSH and checks/starts the Ollama service.
"""

import paramiko
import time
import socket
import logging
import getpass
import sys
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gpu_ollama.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GPUOllama")

def ssh_connect(hostname, username, password, port=22):
    """Connect to the GPU machine via SSH"""
    try:
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Connect to the server
        logger.info(f"Connecting to {hostname} as {username}...")
        client.connect(hostname, port=port, username=username, password=password)
        
        logger.info("Connected successfully!")
        return client
    except paramiko.AuthenticationException:
        logger.error("Authentication failed. Please check your username and password.")
        return None
    except paramiko.SSHException as e:
        logger.error(f"SSH error: {str(e)}")
        return None
    except socket.error as e:
        logger.error(f"Connection error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error connecting to {hostname}: {str(e)}")
        return None

def run_command(client, command, show_output=True):
    """Run a command on the remote machine"""
    try:
        logger.info(f"Running command: {command}")
        stdin, stdout, stderr = client.exec_command(command, timeout=30)
        
        # Get output
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        
        if error and show_output:
            logger.error(f"Command error: {error}")
        
        if output and show_output:
            # Limit output length for logging
            if len(output) > 500:
                logger.info(f"Command output (truncated): {output[:500]}...")
            else:
                logger.info(f"Command output: {output}")
        
        return output, error
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return None, str(e)

def check_ollama_installed(client):
    """Check if Ollama is installed on the GPU machine"""
    # Try multiple ways to check if Ollama is installed
    # First, check using Get-Command
    output, _ = run_command(client, 'powershell -Command "Get-Command ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source"')
    if output:
        return True
    
    # Second, check common installation paths
    paths = [
        'C:\\Program Files\\Ollama\\ollama.exe',
        'C:\\Ollama\\ollama.exe',
        '%USERPROFILE%\\AppData\\Local\\Programs\\Ollama\\ollama.exe'
    ]
    
    for path in paths:
        output, _ = run_command(client, f'powershell -Command "Test-Path \"{path}\""')
        if output and 'True' in output:
            logger.info(f"Found Ollama at {path}")
            return True
    
    return False

def check_ollama_running(client):
    """Check if Ollama service is running"""
    output, _ = run_command(client, 'powershell -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue"')
    return bool(output)

def configure_ollama_for_remote_access(client):
    """Configure Ollama to accept connections from remote machines"""
    # Check if Ollama is already configured for remote access
    run_command(client, 'powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; [Environment]::SetEnvironmentVariable(\'OLLAMA_HOST\', \'0.0.0.0:11434\', \'User\')"')
    
    logger.info("Configured Ollama to listen on all interfaces (0.0.0.0:11434)")
    return True

def start_ollama(client):
    """Start the Ollama service"""
    # First stop any running Ollama process
    run_command(client, 'powershell -Command "Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue"')
    time.sleep(2)
    
    # Start Ollama with the host configuration
    run_command(client, 'powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; Start-Process -FilePath ollama -ArgumentList \'serve\' -WindowStyle Hidden"')
    
    # Wait for service to start
    logger.info("Waiting for Ollama service to start...")
    for i in range(10):
        if check_ollama_running(client):
            logger.info("Ollama service started successfully")
            return True
        logger.info(f"Waiting... ({i+1}/10)")
        time.sleep(2)
    
    logger.error("Failed to start Ollama service")
    return False

def create_ollama_firewall_rule(client):
    """Create a firewall rule to allow Ollama API traffic"""
    # Check if rule already exists
    output, _ = run_command(client, 'powershell -Command "Get-NetFirewallRule -Name ollama-api -ErrorAction SilentlyContinue"')
    
    if output:
        logger.info("Firewall rule for Ollama API already exists")
        return True
    
    # Create the rule
    _, error = run_command(client, 'powershell -Command "New-NetFirewallRule -Name ollama-api -DisplayName \'Ollama API\' -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow"')
    
    if error:
        logger.error(f"Error creating firewall rule: {error}")
        return False
    
    logger.info("Firewall rule created successfully")
    return True

def check_ollama_api(client):
    """Check if the Ollama API is accessible"""
    output, _ = run_command(client, 'powershell -Command "Invoke-RestMethod -Uri http://localhost:11434/api/tags -Method Get -ErrorAction SilentlyContinue"')
    return "models" in output

def get_available_models(client):
    """Get available models from Ollama"""
    output, _ = run_command(client, 'powershell -Command "Invoke-RestMethod -Uri http://localhost:11434/api/tags -Method Get | ConvertTo-Json -Depth 10"')
    
    if not output:
        return []
    
    try:
        import json
        data = json.loads(output)
        return data.get("models", [])
    except:
        logger.error("Error parsing models JSON")
        return []

def main():
    # Connection parameters
    hostname = "10.10.12.202"
    username = "ROWAD\\Omar Essam2"
    password = "PMO@1234"  # Hardcoded password
    
    # Connect to the GPU machine
    client = ssh_connect(hostname, username, password)
    if not client:
        logger.error("Failed to connect to the GPU machine")
        sys.exit(1)
    
    try:
        # Check if Ollama is installed
        if not check_ollama_installed(client):
            logger.error("Ollama is not installed on the GPU machine")
            logger.info("Please install Ollama from https://ollama.com/download/windows")
            sys.exit(1)
        
        # Configure Ollama for remote access
        configure_ollama_for_remote_access(client)
        
        # Check if Ollama is running
        if not check_ollama_running(client):
            logger.info("Ollama is not running. Starting Ollama service...")
            if not start_ollama(client):
                logger.error("Failed to start Ollama service")
                sys.exit(1)
        else:
            logger.info("Ollama service is already running")
            # Restart Ollama to apply the remote access configuration
            logger.info("Restarting Ollama service to apply remote access configuration...")
            if not start_ollama(client):
                logger.error("Failed to restart Ollama service")
                sys.exit(1)
        
        # Create firewall rule
        create_ollama_firewall_rule(client)
        
        # Check if API is accessible
        if not check_ollama_api(client):
            logger.error("Ollama API is not accessible")
            logger.info("Restarting Ollama service...")
            run_command(client, 'powershell -Command "Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue"')
            time.sleep(2)
            if not start_ollama(client):
                logger.error("Failed to restart Ollama service")
                sys.exit(1)
        
        # Get available models
        models = get_available_models(client)
        if models:
            logger.info(f"Available models ({len(models)}):")
            for model in models:
                logger.info(f"  - {model.get('name')}")
        else:
            logger.warning("No models found")
        
        logger.info(f"Ollama API should now be accessible at http://{hostname}:11434")
        logger.info("You can now run your chatbot pointing to this API endpoint")
    
    finally:
        # Close the SSH connection
        client.close()
        logger.info("SSH connection closed")

if __name__ == "__main__":
    main()
