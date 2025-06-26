# Setting up SSH and Ollama on the GPU Machine

This guide will help you set up SSH on the GPU machine and then start the Ollama API service.

## Step 1: Enable SSH on the GPU Machine

1. Log in to the GPU machine (10.10.12.202) directly or via Remote Desktop.

2. Open PowerShell as Administrator and run the following commands:

```powershell
# Check if OpenSSH Server is installed
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'

# Install OpenSSH Server if needed
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0

# Start the SSH service
Start-Service sshd

# Set the service to start automatically
Set-Service -Name sshd -StartupType 'Automatic'

# Confirm the firewall rule is configured
Get-NetFirewallRule -Name *ssh*

# If no firewall rule exists, create one
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

3. Create a user account for SSH access (if needed):

```powershell
# Create a new user (replace USERNAME and PASSWORD with your desired values)
$Password = ConvertTo-SecureString "PASSWORD" -AsPlainText -Force
New-LocalUser -Name "USERNAME" -Password $Password -FullName "SSH User" -Description "User for SSH access"
Add-LocalGroupMember -Group "Administrators" -Member "USERNAME"
```

## Step 2: Test SSH Connection

From your local machine, test the SSH connection:

```powershell
ssh USERNAME@10.10.12.202
```

## Step 3: Start Ollama Service

Once SSH is working, you can start the Ollama service on the GPU machine:

1. Connect to the GPU machine via SSH:

```powershell
ssh USERNAME@10.10.12.202
```

2. Check if Ollama is installed:

```powershell
# For Windows
where ollama

# For Linux
which ollama
```

3. If Ollama is not installed, install it:

```powershell
# For Windows
# Download the installer from https://ollama.com/download/windows and run it

# For Linux
curl -fsSL https://ollama.com/install.sh | sh
```

4. Start the Ollama service:

```powershell
# Start Ollama service
ollama serve
```

5. In a separate terminal, check if the service is running:

```powershell
# Test the Ollama API
curl http://localhost:11434/api/tags
```

## Step 4: Configure Firewall to Allow Remote Access

To allow other machines to access the Ollama API:

```powershell
# Create a firewall rule for Ollama API (port 11434)
New-NetFirewallRule -Name "Ollama API" -DisplayName "Ollama API" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
```

## Step 5: Update Chatbot Configuration

After setting up the Ollama service, update your chatbot configuration to use the correct API URL:

```powershell
python update_chatbot_config.py --chatbot-file "15.vector_chatbot.rev.09(working).py" --api-url "http://10.10.12.202:11434" --test-url
```

## Troubleshooting

If you encounter issues:

1. Check if the Ollama service is running:
```powershell
Get-Process -Name ollama
```

2. Check the Ollama logs:
```powershell
# Location depends on installation method
# Typically in %USERPROFILE%\.ollama\logs
```

3. Test the API locally on the GPU machine:
```powershell
curl http://localhost:11434/api/tags
```

4. Check firewall settings:
```powershell
Get-NetFirewallRule -Name *ollama*
```
