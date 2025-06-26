"""
Chatbot Configuration Updater
This script updates the chatbot configuration to use a different Ollama API URL.
"""

import os
import re
import argparse
import logging
import requests
import time
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_config.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatbotConfig")

def check_ollama_api(api_url, timeout=5):
    """Check if the Ollama API is accessible"""
    try:
        logger.info(f"Checking Ollama API at {api_url}...")
        response = requests.get(f"{api_url}/api/tags", timeout=timeout)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                logger.info(f"Ollama API is accessible. Found {len(models)} models")
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

def update_chatbot_config(chatbot_file, new_api_url, backup=True):
    """
    Update the chatbot configuration to use a different Ollama API URL
    
    Args:
        chatbot_file: Path to the chatbot Python file
        new_api_url: New Ollama API URL to use
        backup: Whether to create a backup of the original file
    """
    try:
        # Check if file exists
        if not os.path.isfile(chatbot_file):
            logger.error(f"Chatbot file not found: {chatbot_file}")
            return False
        
        # Create backup if requested
        if backup:
            backup_file = f"{chatbot_file}.bak"
            logger.info(f"Creating backup of chatbot file: {backup_file}")
            shutil.copy2(chatbot_file, backup_file)
        
        # Read the file
        with open(chatbot_file, 'r') as f:
            content = f.read()
        
        # Update the Ollama API URL
        # Pattern to match the OLLAMA_API_URL definition
        pattern = r'(OLLAMA_API_URL\s*=\s*)["\']([^"\']+)["\']'
        
        if re.search(pattern, content):
            # Replace the URL
            updated_content = re.sub(pattern, f'\\1"{new_api_url}"', content)
            
            # Write the updated content back to the file
            with open(chatbot_file, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Updated Ollama API URL to {new_api_url}")
            return True
        else:
            logger.error("Could not find OLLAMA_API_URL in the chatbot file")
            return False
    
    except Exception as e:
        logger.error(f"Error updating chatbot config: {str(e)}")
        return False

def update_alternative_urls(chatbot_file, urls, backup=True):
    """
    Update the alternative Ollama API URLs in the chatbot file
    
    Args:
        chatbot_file: Path to the chatbot Python file
        urls: List of alternative URLs to use
        backup: Whether to create a backup of the original file
    """
    try:
        # Check if file exists
        if not os.path.isfile(chatbot_file):
            logger.error(f"Chatbot file not found: {chatbot_file}")
            return False
        
        # Create backup if requested
        if backup and not os.path.exists(f"{chatbot_file}.bak"):
            backup_file = f"{chatbot_file}.bak"
            logger.info(f"Creating backup of chatbot file: {backup_file}")
            shutil.copy2(chatbot_file, backup_file)
        
        # Read the file
        with open(chatbot_file, 'r') as f:
            content = f.read()
        
        # Format the URLs list as a Python list
        urls_str = "[\n"
        for url in urls:
            urls_str += f'    "{url}",  # Alternative URL\n'
        urls_str += "]"
        
        # Pattern to match the ALTERNATIVE_OLLAMA_URLS definition
        pattern = r'ALTERNATIVE_OLLAMA_URLS\s*=\s*\[([\s\S]*?)\]'
        
        if re.search(pattern, content):
            # Replace the URLs list
            updated_content = re.sub(pattern, f'ALTERNATIVE_OLLAMA_URLS = {urls_str}', content)
            
            # Write the updated content back to the file
            with open(chatbot_file, 'w') as f:
                f.write(updated_content)
            
            logger.info(f"Updated alternative Ollama API URLs")
            return True
        else:
            logger.error("Could not find ALTERNATIVE_OLLAMA_URLS in the chatbot file")
            return False
    
    except Exception as e:
        logger.error(f"Error updating alternative URLs: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update chatbot configuration")
    parser.add_argument("--chatbot-file", required=True, help="Path to the chatbot Python file")
    parser.add_argument("--api-url", help="New Ollama API URL to use")
    parser.add_argument("--add-alternative", action="append", help="Add an alternative URL (can be used multiple times)")
    parser.add_argument("--test-url", action="store_true", help="Test the new API URL before updating")
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original file")
    
    args = parser.parse_args()
    
    # Test the new API URL if requested
    if args.api_url and args.test_url:
        api_accessible, _ = check_ollama_api(args.api_url)
        if not api_accessible:
            logger.error(f"New API URL {args.api_url} is not accessible. Aborting update.")
            return
    
    # Update the main API URL if provided
    if args.api_url:
        success = update_chatbot_config(args.chatbot_file, args.api_url, not args.no_backup)
        if not success:
            logger.error("Failed to update chatbot configuration")
            return
    
    # Update alternative URLs if provided
    if args.add_alternative:
        # First, get the current alternative URLs
        with open(args.chatbot_file, 'r') as f:
            content = f.read()
        
        # Extract current alternative URLs
        pattern = r'ALTERNATIVE_OLLAMA_URLS\s*=\s*\[([\s\S]*?)\]'
        match = re.search(pattern, content)
        
        current_urls = []
        if match:
            # Extract URLs from the match
            urls_text = match.group(1)
            url_pattern = r'"([^"]+)"'
            current_urls = re.findall(url_pattern, urls_text)
        
        # Add new alternative URLs
        new_urls = current_urls.copy()
        for url in args.add_alternative:
            if url not in new_urls:
                new_urls.append(url)
        
        # Update the file with new URLs
        success = update_alternative_urls(args.chatbot_file, new_urls, not args.no_backup)
        if not success:
            logger.error("Failed to update alternative URLs")
            return
    
    logger.info("Chatbot configuration updated successfully")

if __name__ == "__main__":
    main()
