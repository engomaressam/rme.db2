import paramiko

# SSH details
host = "10.10.11.242"
username = "PMO"
password = "PMO@1234"
sudo_password = "PMO@1234"

# Commands to run on the server
commands = [
    "cd ~/airflow",
    "source airflow/venv/bin/activate && airflow db migrate",
    "source airflow/venv/bin/activate && airflow webserver -D",
    "source airflow/venv/bin/activate && airflow scheduler -D",
    f"echo {sudo_password} | sudo -S /opt/spark/sbin/start-master.sh -i 10.10.11.242 --webui-port 8080",
    f"echo {sudo_password} | sudo -S /opt/spark/sbin/start-worker.sh spark://10.10.11.242:7077",
    f"echo {sudo_password} | sudo -S jps"
]


def execute_ssh_commands(host, username, password, commands):
    try:
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the server
        print(f"Connecting to {host}...")
        client.connect(hostname=host, username=username, password=password)

        # Execute each command
        for command in commands:
            print(f"Executing: {command}")
            stdin, stdout, stderr = client.exec_command(command)
            print(stdout.read().decode())
            print(stderr.read().decode())

        # Close the connection
        client.close()
        print("All commands executed successfully.")

    except Exception as e:
        print(f"Error: {e}")

# Run the function
execute_ssh_commands(host, username, password, commands)
