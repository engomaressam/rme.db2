"""
Simple Ollama Fix
A simplified script to fix the Ollama connection issues.
"""

import paramiko
import time
import sys

def main():
    # Connection parameters
    hostname = "10.10.12.202"
    username = "ROWAD\\Omar Essam2"
    password = "PMO@1234"
    
    try:
        print("Connecting to GPU machine...")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname, port=22, username=username, password=password)
        print("Connected successfully!")
        
        # Execute commands one by one with clear output
        commands = [
            ("Stopping Ollama...", 'powershell -Command "Stop-Process -Name ollama -Force -ErrorAction SilentlyContinue"'),
            ("Setting environment variable...", 'powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; [Environment]::SetEnvironmentVariable(\'OLLAMA_HOST\', \'0.0.0.0:11434\', \'User\')"'),
            ("Starting Ollama...", 'powershell -Command "$env:OLLAMA_HOST=\'0.0.0.0:11434\'; Start-Process -FilePath ollama -ArgumentList \'serve\' -WindowStyle Hidden"'),
            ("Creating firewall rule...", 'powershell -Command "Get-NetFirewallRule -Name ollama-api -ErrorAction SilentlyContinue || New-NetFirewallRule -Name ollama-api -DisplayName \'Ollama API\' -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow"')
        ]
        
        for desc, cmd in commands:
            print(f"\n{desc}")
            stdin, stdout, stderr = client.exec_command(cmd)
            time.sleep(2)  # Give each command time to complete
        
        # Wait for Ollama to start
        print("\nWaiting for Ollama to start...")
        time.sleep(5)
        
        # Check if Ollama is running
        print("\nChecking if Ollama is running...")
        stdin, stdout, stderr = client.exec_command('powershell -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue | Select-Object Id, Name"')
        output = stdout.read().decode().strip()
        print(output if output else "Ollama process not found")
        
        # Close the connection
        client.close()
        print("\nSSH connection closed")
        print("\nOllama should now be accessible at http://10.10.12.202:11434")
        print("You can now run your chatbot with this API URL")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
