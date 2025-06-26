"""
Alstom Project Assistant - Simple Public Access Setup
This script sets up a public URL for the Alstom Project Assistant using ngrok.
It assumes ngrok is already installed at C:\Program Files\ngrok-v3-stable-windows-amd64
"""

import os
import sys
import subprocess
import time
import requests
import webbrowser

# Configuration
CHATBOT_SCRIPT = "15.vector_chatbot.rev.09(working).py"  # The working chatbot script
CHATBOT_PORT = 7860  # Default Gradio port
NGROK_PATH = r"C:\Program Files\ngrok-v3-stable-windows-amd64\ngrok.exe"
NGROK_AUTH_TOKEN = "2xS34DIbllsdySS37K94bRF2Fem_4nYW2qaP9C1ndeUhMZtcu"  # Your ngrok auth token

def install_dependencies():
    """Install all required dependencies"""
    print("Installing required dependencies...")
    dependencies = [
        "gradio",
        "requests",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "pyngrok"
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("All dependencies installed successfully!")

def setup_ngrok_auth():
    """Set up ngrok authentication"""
    print(f"Setting up ngrok auth token: {NGROK_AUTH_TOKEN[:5]}...")
    subprocess.run([NGROK_PATH, "config", "add-authtoken", NGROK_AUTH_TOKEN], 
                  stdout=subprocess.PIPE, 
                  stderr=subprocess.PIPE)
    print("ngrok auth token configured successfully")

def start_chatbot():
    """Start the chatbot in a separate process"""
    print(f"Starting chatbot: {CHATBOT_SCRIPT}")
    
    # Get the absolute path to the chatbot script
    script_path = os.path.abspath(CHATBOT_SCRIPT)
    
    # Start the chatbot process
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(f"Chatbot process started with PID: {process.pid}")
    return process

def wait_for_chatbot(timeout=60):
    """Wait for the chatbot to be ready by checking the local URL"""
    print(f"Waiting for chatbot to be ready (timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{CHATBOT_PORT}", timeout=2)
            if response.status_code == 200:
                print("Chatbot is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f"\nTimed out waiting for chatbot to be ready after {timeout}s")
    return False

def start_ngrok_tunnel():
    """Start an ngrok tunnel to the chatbot port"""
    print(f"Starting ngrok tunnel to port {CHATBOT_PORT}...")
    
    # Start ngrok process
    process = subprocess.Popen(
        [NGROK_PATH, "http", str(CHATBOT_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a moment for ngrok to start
    time.sleep(5)
    
    # Get the public URL from the ngrok API
    try:
        response = requests.get("http://localhost:4040/api/tunnels")
        tunnels = response.json()["tunnels"]
        
        if tunnels:
            public_url = tunnels[0]["public_url"]
            print(f"ngrok tunnel established: {public_url}")
            return public_url, process
        else:
            print("No ngrok tunnels found")
            return None, process
    except Exception as e:
        print(f"Error getting ngrok tunnel URL: {str(e)}")
        return None, process

def main():
    print("="*50)
    print("Alstom Project Assistant - Public Access Setup")
    print("="*50)
    
    # Check if dependencies should be installed
    install_deps = input("Do you want to install all required dependencies? (y/n): ").strip().lower()
    if install_deps == 'y':
        install_dependencies()
    
    # Set up ngrok authentication
    setup_ngrok_auth()
    
    # Start the chatbot
    chatbot_process = start_chatbot()
    
    # Wait for the chatbot to be ready
    wait_for_chatbot(timeout=60)
    
    # Start the ngrok tunnel
    public_url, ngrok_process = start_ngrok_tunnel()
    
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
        
        if ngrok_process and ngrok_process.poll() is None:
            print("Terminating ngrok process...")
            ngrok_process.terminate()
            ngrok_process.wait(timeout=5)
    
    print("Public access has been closed.")

if __name__ == "__main__":
    main()
