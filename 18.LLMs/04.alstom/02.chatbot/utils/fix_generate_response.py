"""
Fix for the generate_response function in the Alstom chatbot
"""

import os
import sys
import json
import time
import requests
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chatbot_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatbotFix")

# Ollama API configuration
OLLAMA_API_URL = "http://10.10.12.202:11434"
MODEL_NAME = "qwen2.5-coder:7b"

def fixed_generate_response(prompt, system_prompt=None):
    """
    Fixed version of the generate_response function
    This includes better error handling and logging
    """
    try:
        logger.info(f"Generating response using {MODEL_NAME}")
        logger.info(f"API URL: {OLLAMA_API_URL}")
        
        # Prepare request data
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            data["system"] = system_prompt
            logger.info(f"Using system prompt: {system_prompt}")
        
        # Log the request data (excluding potentially large prompt)
        logger.info(f"Request data: {json.dumps({k: v for k, v in data.items() if k != 'prompt'})}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Send request with longer timeout
        logger.info("Sending request to Ollama API...")
        start_time = time.time()
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate", 
            json=data,
            timeout=120  # Much longer timeout
        )
        end_time = time.time()
        
        # Process response
        logger.info(f"Response received in {end_time - start_time:.2f} seconds")
        logger.info(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', 'No response')
            logger.info(f"Response length: {len(response_text)} characters")
            return response_text
        else:
            logger.error(f"Error: {response.status_code}")
            logger.error(response.text)
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        logger.exception(f"Exception in generate_response: {e}")
        return f"Error: {str(e)}"

# Instructions for using this fix
print("""
To fix the chatbot, replace the generate_response function in 15.vector_chatbot.rev.10.py
with the fixed_generate_response function from this file.

You can also add these lines to the top of the process_question function:

try:
    # Your existing code
    ...
except Exception as e:
    logger.exception(f"Exception in process_question: {e}")
    return history + [(message, f"Error: {str(e)}")], doc_paths, ""
""")

# Test the fixed function
if __name__ == "__main__":
    print(f"Testing fixed generate_response function with Ollama API at {OLLAMA_API_URL}")
    
    # Test with a simple prompt
    test_prompt = "What is the capital of France?"
    test_system = "You are a helpful assistant."
    
    print("\nTesting with simple prompt...")
    response = fixed_generate_response(test_prompt, test_system)
    print("\nResponse:")
    print(response)
