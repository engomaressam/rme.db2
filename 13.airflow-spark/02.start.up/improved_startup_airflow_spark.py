import paramiko
import time

# SSH details
host = "10.10.11.242"
username = "PMO"
password = "PMO@1234"
sudo_password = "PMO@1234"

# Function to check if a process is running
def check_process_running(client, process_name):
    """Check if a process is running using ps command"""
    stdin, stdout, stderr = client.exec_command(f"ps -ef | grep {process_name} | grep -v grep")
    result = stdout.read().decode()
    return len(result) > 0

# Function to kill a process by PID
def kill_process_by_pid(client, pid_file, process_name):
    """Kill a process using its PID file if it exists"""
    try:
        # Check if PID file exists
        stdin, stdout, stderr = client.exec_command(f"test -f {pid_file} && echo 'exists' || echo 'not exists'")
        if 'exists' in stdout.read().decode():
            # Read PID from file
            stdin, stdout, stderr = client.exec_command(f"cat {pid_file}")
            pid = stdout.read().decode().strip()
            if pid:
                print(f"Found {process_name} running with PID {pid}, stopping it...")
                # Kill the process
                stdin, stdout, stderr = client.exec_command(f"echo {sudo_password} | sudo -S kill -9 {pid}")
                time.sleep(2)  # Give it time to stop
                # Remove PID file
                stdin, stdout, stderr = client.exec_command(f"echo {sudo_password} | sudo -S rm {pid_file}")
                return True
    except Exception as e:
        print(f"Error stopping {process_name}: {e}")
    return False

# Commands to run on the server with proper cleanup
def execute_ssh_commands(host, username, password):
    try:
        # Create SSH client
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the server
        print(f"Connecting to {host}...")
        client.connect(hostname=host, username=username, password=password)
        
        # Change to airflow directory
        print("Changing to airflow directory...")
        stdin, stdout, stderr = client.exec_command("cd ~/airflow")
        
        # Stop existing Airflow services if running
        print("Checking for existing Airflow services...")
        
        # Stop scheduler if running
        kill_process_by_pid(client, "/home/PMO/airflow/airflow-scheduler.pid", "airflow scheduler")
        
        # Stop webserver if running
        kill_process_by_pid(client, "/home/PMO/airflow/airflow-webserver.pid", "airflow webserver")
        kill_process_by_pid(client, "/home/PMO/airflow/airflow-webserver-monitor.pid", "airflow webserver")
        
        # Check for Spark processes and stop them if needed
        print("Checking for existing Spark services...")
        if check_process_running(client, "org.apache.spark.deploy.master.Master"):
            print("Stopping Spark Master...")
            stdin, stdout, stderr = client.exec_command(f"echo {sudo_password} | sudo -S /opt/spark/sbin/stop-master.sh")
            time.sleep(2)
            
        if check_process_running(client, "org.apache.spark.deploy.worker.Worker"):
            print("Stopping Spark Worker...")
            stdin, stdout, stderr = client.exec_command(f"echo {sudo_password} | sudo -S /opt/spark/sbin/stop-worker.sh")
            time.sleep(2)
        
        # Start Airflow services
        print("Starting Airflow services...")
        
        # Run database migrations
        print("Running Airflow database migrations...")
        stdin, stdout, stderr = client.exec_command("cd ~/airflow && source airflow/venv/bin/activate && airflow db migrate")
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        if stdout_result:
            print(f"DB Migration Output: {stdout_result}")
        if stderr_result:
            print(f"DB Migration Error: {stderr_result}")
        
        # Start scheduler
        print("Starting Airflow scheduler...")
        stdin, stdout, stderr = client.exec_command("cd ~/airflow && source airflow/venv/bin/activate && airflow scheduler -D")
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        if stdout_result:
            print(f"Scheduler Output: {stdout_result}")
        if stderr_result:
            print(f"Scheduler Error: {stderr_result}")
        
        # Start webserver
        print("Starting Airflow webserver...")
        stdin, stdout, stderr = client.exec_command("cd ~/airflow && source airflow/venv/bin/activate && airflow webserver -D --port 8090")
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        if stdout_result:
            print(f"Webserver Output: {stdout_result}")
        if stderr_result:
            print(f"Webserver Error: {stderr_result}")
        
        # Start Spark services
        print("Starting Spark services...")
        
        # Start Spark master
        print("Starting Spark master...")
        stdin, stdout, stderr = client.exec_command(f"echo {sudo_password} | sudo -S /opt/spark/sbin/start-master.sh -i 10.10.11.242 --webui-port 8080")
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        if stdout_result:
            print(f"Spark Master Output: {stdout_result}")
        if stderr_result and "[sudo] password for PMO:" not in stderr_result:
            print(f"Spark Master Error: {stderr_result}")
        
        # Start Spark worker
        print("Starting Spark worker...")
        stdin, stdout, stderr = client.exec_command(f"echo {sudo_password} | sudo -S /opt/spark/sbin/start-worker.sh spark://10.10.11.242:7077")
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        if stdout_result:
            print(f"Spark Worker Output: {stdout_result}")
        if stderr_result and "[sudo] password for PMO:" not in stderr_result:
            print(f"Spark Worker Error: {stderr_result}")
        
        # Verify running processes
        print("Verifying running processes...")
        stdin, stdout, stderr = client.exec_command(f"echo {sudo_password} | sudo -S jps")
        stdout_result = stdout.read().decode()
        stderr_result = stderr.read().decode()
        if stdout_result:
            print(f"JPS Output: {stdout_result}")
        if stderr_result and "[sudo] password for PMO:" not in stderr_result:
            print(f"JPS Error: {stderr_result}")
        
        # Alternative way to check running processes
        print("Checking running processes with ps...")
        stdin, stdout, stderr = client.exec_command("ps -ef | grep -E 'airflow|spark' | grep -v grep")
        stdout_result = stdout.read().decode()
        if stdout_result:
            print(f"Running processes:\n{stdout_result}")
        
        # Close the connection
        client.close()
        print("All commands executed successfully.")

    except Exception as e:
        print(f"Error: {e}")

# Run the function
if __name__ == "__main__":
    execute_ssh_commands(host, username, password)
