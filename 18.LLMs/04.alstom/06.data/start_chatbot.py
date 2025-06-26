"""
Alstom Chatbot Starter
This script updates the chatbot configuration and starts it with the correct Ollama API URL.
"""

import os
import sys
import subprocess
import time
import requests
import logging
import argparse
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_starter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatbotStarter")

def check_ollama_api(api_url, timeout=5):
    """Check if the Ollama API is accessible and get available models"""
    try:
        logger.info(f"Checking Ollama API at {api_url}...")
        response = requests.get(f"{api_url}/api/tags", timeout=timeout)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                logger.info(f"Ollama API is accessible. Found {len(models)} models:")
                for model in models:
                    logger.info(f"  - {model.get('name')}")
                return True, models
            else:
                logger.warning("Ollama API is accessible but no models found")
                return True, []
        else:
            logger.error(f"Ollama API returned status code {response.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to Ollama API at {api_url}")
        return False, []
    except Exception as e:
        logger.error(f"Error checking Ollama API: {str(e)}")
        return False, []

def update_available_models(chatbot_file, models):
    """Update the AVAILABLE_MODELS dictionary in the chatbot file"""
    try:
        # Create a formatted models dictionary
        models_dict = {}
        for model in models:
            model_name = model.get('name')
            display_name = model_name
            
            # Create a more user-friendly display name
            if 'qwen' in model_name.lower():
                display_name = f"Qwen ({model_name.split(':')[1] if ':' in model_name else model_name})"
            elif 'llama' in model_name.lower():
                display_name = f"Llama ({model_name.split(':')[1] if ':' in model_name else model_name})"
            elif 'mistral' in model_name.lower():
                display_name = f"Mistral ({model_name.split(':')[1] if ':' in model_name else model_name})"
            elif 'deepseek' in model_name.lower():
                display_name = f"DeepSeek ({model_name.split(':')[1] if ':' in model_name else model_name})"
            
            models_dict[display_name] = model_name
        
        # Format the dictionary as a string
        models_str = "{\n"
        for display_name, model_name in models_dict.items():
            models_str += f'    "{display_name}": "{model_name}",\n'
        models_str += "}"
        
        # Read the chatbot file
        with open(chatbot_file, 'r') as f:
            content = f.read()
        
        # Find the AVAILABLE_MODELS dictionary
        import re
        pattern = r'AVAILABLE_MODELS\s*=\s*{[^}]*}'
        
        if re.search(pattern, content):
            # Replace the dictionary
            updated_content = re.sub(pattern, f'AVAILABLE_MODELS = {models_str}', content)
            
            # Write the updated content back to the file
            with open(chatbot_file, 'w') as f:
                f.write(updated_content)
            
            logger.info("Updated available models in chatbot file")
            return True
        else:
            logger.error("Could not find AVAILABLE_MODELS in the chatbot file")
            return False
    except Exception as e:
        logger.error(f"Error updating available models: {str(e)}")
        return False

def start_chatbot(chatbot_file, api_url):
    """Start the chatbot with the specified API URL"""
    try:
        # First, update the chatbot configuration
        update_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "update_chatbot_config.py")
        
        if os.path.isfile(update_script):
            logger.info(f"Updating chatbot configuration to use API URL: {api_url}")
            
            subprocess.run([
                sys.executable,
                update_script,
                "--chatbot-file", chatbot_file,
                "--api-url", api_url,
                "--test-url"
            ])
        
        # Start the chatbot
        logger.info("Starting chatbot...")
        subprocess.Popen([
            sys.executable,
            chatbot_file
        ])
        
        logger.info("Chatbot started successfully")
        return True
    except Exception as e:
        logger.error(f"Error starting chatbot: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Start Alstom Chatbot with the correct Ollama API URL")
    parser.add_argument("--chatbot-file", default="15.vector_chatbot.rev.09(working).py", help="Path to the chatbot Python file")
    parser.add_argument("--api-url", default="http://10.10.12.202:11434", help="Ollama API URL to use")
    parser.add_argument("--update-models", action="store_true", help="Update available models in the chatbot file")
    
    args = parser.parse_args()
    
    # Check if the chatbot file exists
    if not os.path.isfile(args.chatbot_file):
        logger.error(f"Chatbot file not found: {args.chatbot_file}")
        sys.exit(1)
    
    # Check if the Ollama API is accessible
    api_accessible, models = check_ollama_api(args.api_url)
    
    if not api_accessible:
        logger.error(f"Ollama API at {args.api_url} is not accessible")
        
        # Try alternative URLs
        alternative_urls = [
            "http://localhost:11434",
            "http://127.0.0.1:11434"
        ]
        
        for alt_url in alternative_urls:
            logger.info(f"Trying alternative URL: {alt_url}")
            alt_accessible, alt_models = check_ollama_api(alt_url)
            
            if alt_accessible:
                logger.info(f"Alternative URL {alt_url} is accessible")
                args.api_url = alt_url
                models = alt_models
                api_accessible = True
                break
        
        if not api_accessible:
            logger.error("No accessible Ollama API found")
            sys.exit(1)
    
    # Update available models if requested
    if args.update_models and models:
        update_available_models(args.chatbot_file, models)
    
    # Start the chatbot
    start_chatbot(args.chatbot_file, args.api_url)

if __name__ == "__main__":
    main()
