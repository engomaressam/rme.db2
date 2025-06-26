"""
Simple Chatbot Updater
This script updates the chatbot to use the Ollama API on the GPU machine.
"""

import os
import re
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatbotUpdater")

def update_chatbot_config(chatbot_file, api_url):
    """Update the chatbot configuration to use the specified Ollama API URL"""
    try:
        # Check if file exists
        if not os.path.isfile(chatbot_file):
            logger.error(f"Chatbot file not found: {chatbot_file}")
            return False
        
        # Create backup
        backup_file = f"{chatbot_file}.bak"
        logger.info(f"Creating backup of chatbot file: {backup_file}")
        with open(chatbot_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
        
        # Read the file
        with open(chatbot_file, 'r') as f:
            content = f.read()
        
        # Update the Ollama API URL
        pattern = r'(OLLAMA_API_URL\s*=\s*)["\']([^"\']+)["\']'
        
        if re.search(pattern, content):
            # Replace the URL
            updated_content = re.sub(pattern, f'\\1"{api_url}"', content)
            
            # Write the updated content back to the file
            with open(chatbot_file, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Updated Ollama API URL to {api_url}")
            return True
        else:
            logger.error("Could not find OLLAMA_API_URL in the chatbot file")
            return False
    
    except Exception as e:
        logger.error(f"Error updating chatbot config: {str(e)}")
        return False

def main():
    # Configuration
    chatbot_file = "15.vector_chatbot.rev.09(working).py"
    api_url = "http://10.10.12.202:11434"
    
    # Update the chatbot configuration
    if update_chatbot_config(chatbot_file, api_url):
        logger.info("Chatbot configuration updated successfully")
        logger.info(f"You can now run the chatbot with: python {chatbot_file}")
    else:
        logger.error("Failed to update chatbot configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
