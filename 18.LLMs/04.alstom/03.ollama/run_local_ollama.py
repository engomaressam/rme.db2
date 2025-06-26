"""
Local Ollama Service Runner
This script helps set up and run Ollama locally as a fallback option.
"""

import os
import sys
import subprocess
import time
import requests
import logging
import argparse
import platform
import webbrowser
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("local_ollama.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LocalOllama")

def check_ollama_installed():
    """Check if Ollama is installed on the system"""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking Ollama installation: {str(e)}")
        return False

def download_ollama():
    """Open the Ollama download page in a browser"""
    download_url = "https://ollama.com/download"
    logger.info(f"Opening Ollama download page: {download_url}")
    webbrowser.open(download_url)
    
    logger.info("Please download and install Ollama, then run this script again.")
    return False

def check_ollama_running():
    """Check if Ollama service is already running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_ollama_service():
    """Start the Ollama service"""
    try:
        logger.info("Starting Ollama service...")
        
        if platform.system() == "Windows":
            # On Windows, start in a new window to keep it running
            process = subprocess.Popen(
                ["start", "cmd", "/k", "ollama", "serve"], 
                shell=True
            )
        else:
            # On Unix systems, start in background
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        
        # Wait for service to start
        logger.info("Waiting for Ollama service to start...")
        for _ in range(10):
            if check_ollama_running():
                logger.info("Ollama service started successfully")
                return True
            time.sleep(1)
        
        logger.warning("Ollama service did not start within the expected time")
        return False
    except Exception as e:
        logger.error(f"Error starting Ollama service: {str(e)}")
        return False

def pull_model(model_name):
    """Pull a model from Ollama library"""
    try:
        logger.info(f"Pulling model {model_name}...")
        
        # Check if model already exists
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if model.get("name") == model_name:
                    logger.info(f"Model {model_name} is already available")
                    return True
        
        # Pull the model
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Show progress
        logger.info(f"Pulling model {model_name}. This may take a while...")
        
        # Read output line by line to show progress
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(line)
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Successfully pulled model {model_name}")
            return True
        else:
            error = process.stderr.read()
            logger.error(f"Failed to pull model {model_name}: {error}")
            return False
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {str(e)}")
        return False

def update_chatbot_config(chatbot_file):
    """Update the chatbot configuration to use the local Ollama API"""
    try:
        # Check if the update_chatbot_config.py script exists
        update_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_chatbot_config.py")
        
        if os.path.isfile(update_script):
            logger.info("Updating chatbot configuration to use local Ollama API...")
            
            subprocess.run([
                sys.executable,
                update_script,
                "--chatbot-file", chatbot_file,
                "--api-url", "http://localhost:11434",
                "--test-url"
            ])
            
            logger.info("Chatbot configuration updated successfully")
            return True
        else:
            logger.error(f"Update script not found: {update_script}")
            return False
    except Exception as e:
        logger.error(f"Error updating chatbot configuration: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Ollama locally")
    parser.add_argument("--pull", help="Pull a specific model after starting Ollama")
    parser.add_argument("--chatbot-file", help="Path to the chatbot Python file to update")
    
    args = parser.parse_args()
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        logger.error("Ollama is not installed")
        download_ollama()
        return
    
    # Check if Ollama is already running
    if check_ollama_running():
        logger.info("Ollama service is already running")
    else:
        # Start Ollama service
        if not start_ollama_service():
            logger.error("Failed to start Ollama service")
            return
    
    # Pull model if specified
    if args.pull:
        pull_model(args.pull)
    
    # Update chatbot configuration if specified
    if args.chatbot_file:
        update_chatbot_config(args.chatbot_file)
    
    # List available models
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info(f"Available models ({len(models)}):")
            for model in models:
                logger.info(f"  - {model.get('name')}")
        else:
            logger.error("Failed to get available models")
    except Exception as e:
        logger.error(f"Error getting available models: {str(e)}")
    
    logger.info("\nLocal Ollama service is running at http://localhost:11434")
    logger.info("You can now run your chatbot pointing to this API endpoint")

if __name__ == "__main__":
    main()
