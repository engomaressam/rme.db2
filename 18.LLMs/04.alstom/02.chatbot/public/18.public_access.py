"""
Alstom Project Assistant - Public Access Setup
This script sets up a public URL for the Alstom Project Assistant using ngrok.
It allows managers and team members to access the chatbot from anywhere.
"""

import os
import sys
import subprocess
import time
import requests
import json
import logging
import threading
import webbrowser
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("public_access.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PublicAccess")

# Configuration
CHATBOT_SCRIPT = "15.vector_chatbot.rev.09(working).py"  # The working chatbot script
CHATBOT_PORT = 7860  # Default Gradio port
NGROK_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".ngrok2")
NGROK_TIMEOUT = 60  # Seconds to wait for ngrok to start

def check_ngrok_installed():
    """Check if ngrok is installed and available in PATH"""
    try:
        result = subprocess.run(["ngrok", "--version"], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True, 
                               check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_ngrok():
    """Install ngrok using pip and download the binary manually"""
    logger.info("Installing pyngrok via pip...")
    try:
        # Install pyngrok package
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        logger.info("pyngrok installed successfully")
        
        # Create a temporary directory for downloading ngrok
        import tempfile
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, "ngrok.zip")
        
        # Download ngrok zip file
        logger.info("Downloading ngrok binary...")
        ngrok_url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
        
        import urllib.request
        urllib.request.urlretrieve(ngrok_url, zip_path)
        
        # Extract the zip file
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract to user's home directory
            extract_dir = os.path.join(os.path.expanduser("~"), "ngrok")
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)
        
        # Add to PATH temporarily
        os.environ["PATH"] = extract_dir + os.pathsep + os.environ["PATH"]
        
        logger.info(f"ngrok installed successfully to {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to install ngrok: {str(e)}")
        return False

def setup_ngrok_auth(auth_token=None):
    """Set up ngrok authentication"""
    if auth_token is None:
        # Check if auth token is already configured
        config_path = os.path.join(NGROK_CONFIG_DIR, "ngrok.yml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if "authtoken" in f.read():
                    logger.info("ngrok auth token already configured")
                    return True
        
        # Ask for auth token
        print("\n" + "="*50)
        print("ngrok requires an auth token to create persistent URLs")
        print("You can get a free auth token by signing up at https://ngrok.com")
        print("="*50)
        auth_token = input("Enter your ngrok auth token (or press Enter to skip): ").strip()
        
        if not auth_token:
            logger.warning("No auth token provided. ngrok will use a random URL that changes each time.")
            return False
    
    # Set up auth token
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(auth_token)
        logger.info("ngrok auth token configured successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to set ngrok auth token: {str(e)}")
        return False

def start_chatbot():
    """Start the chatbot in a separate process"""
    logger.info(f"Starting chatbot: {CHATBOT_SCRIPT}")
    
    # Get the absolute path to the chatbot script
    script_path = os.path.abspath(CHATBOT_SCRIPT)
    
    # Start the chatbot process
    try:
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        logger.info(f"Chatbot process started with PID: {process.pid}")
        return process
    except Exception as e:
        logger.error(f"Failed to start chatbot: {str(e)}")
        return None

def start_ngrok_tunnel(port=CHATBOT_PORT):
    """Start an ngrok tunnel to the specified port"""
    try:
        from pyngrok import ngrok
        
        # Check for any existing tunnels and close them
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            ngrok.disconnect(tunnel.public_url)
        
        # Start a new tunnel
        logger.info(f"Starting ngrok tunnel to port {port}...")
        http_tunnel = ngrok.connect(port, "http")
        logger.info(f"ngrok tunnel established: {http_tunnel.public_url}")
        
        # Return the public URL
        return http_tunnel.public_url
    except Exception as e:
        logger.error(f"Failed to start ngrok tunnel: {str(e)}")
        return None

def wait_for_chatbot(timeout=30):
    """Wait for the chatbot to be ready by checking the local URL"""
    logger.info(f"Waiting for chatbot to be ready (timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{CHATBOT_PORT}", timeout=2)
            if response.status_code == 200:
                logger.info("Chatbot is ready")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
    
    logger.warning(f"Timed out waiting for chatbot to be ready after {timeout}s")
    return False

def monitor_process(process):
    """Monitor the chatbot process and log its output"""
    while True:
        output = process.stdout.readline()
        if output:
            logger.info(f"Chatbot: {output.strip()}")
        
        error = process.stderr.readline()
        if error:
            logger.error(f"Chatbot error: {error.strip()}")
        
        # Check if process has terminated
        if process.poll() is not None:
            remaining_output, remaining_error = process.communicate()
            if remaining_output:
                logger.info(f"Chatbot final output: {remaining_output.strip()}")
            if remaining_error:
                logger.error(f"Chatbot final error: {remaining_error.strip()}")
            
            logger.info(f"Chatbot process exited with code: {process.returncode}")
            break
        
        time.sleep(0.1)

def main():
    print("="*50)
    print("Alstom Project Assistant - Public Access Setup")
    print("="*50)
    
    # Check if ngrok is installed
    if not check_ngrok_installed():
        print("ngrok is not installed. Installing now...")
        if not install_ngrok():
            print("Failed to install ngrok. Please install it manually.")
            print("You can download it from: https://ngrok.com/download")
            return
    
    # Set up ngrok authentication
    setup_ngrok_auth()
    
    # Start the chatbot
    chatbot_process = start_chatbot()
    if not chatbot_process:
        print("Failed to start the chatbot. Exiting.")
        return
    
    # Start monitoring the chatbot process in a separate thread
    monitor_thread = threading.Thread(target=monitor_process, args=(chatbot_process,))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Wait for the chatbot to be ready
    if not wait_for_chatbot(timeout=60):
        print("Chatbot is taking too long to start. The ngrok tunnel will be created anyway.")
    
    # Start the ngrok tunnel
    public_url = start_ngrok_tunnel(CHATBOT_PORT)
    if not public_url:
        print("Failed to create ngrok tunnel. Exiting.")
        chatbot_process.terminate()
        return
    
    # Display the public URL
    print("\n" + "="*50)
    print("Alstom Project Assistant is now accessible online!")
    print("="*50)
    print(f"Public URL: {public_url}")
    print("\nShare this URL with managers and team members to access the chatbot.")
    print("The URL will remain active as long as this script is running.")
    print("Press Ctrl+C to stop the server and close the public access.")
    print("="*50)
    
    # Open the URL in the default browser
    webbrowser.open(public_url)
    
    # Keep the script running until interrupted
    try:
        while True:
            time.sleep(1)
            
            # Check if the chatbot process is still running
            if chatbot_process.poll() is not None:
                print("Chatbot process has terminated. Shutting down.")
                break
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up
        if chatbot_process and chatbot_process.poll() is None:
            print("Terminating chatbot process...")
            chatbot_process.terminate()
            chatbot_process.wait(timeout=5)
        
        # Disconnect ngrok tunnels
        try:
            from pyngrok import ngrok
            tunnels = ngrok.get_tunnels()
            for tunnel in tunnels:
                ngrok.disconnect(tunnel.public_url)
            print("ngrok tunnels disconnected")
        except Exception as e:
            print(f"Error disconnecting ngrok tunnels: {str(e)}")
    
    print("Public access has been closed.")

if __name__ == "__main__":
    main()
