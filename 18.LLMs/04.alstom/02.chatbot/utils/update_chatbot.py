"""
Script to update the generate_response function in the Alstom chatbot
"""
import os
import re
import shutil

# Path to the chatbot file
CHATBOT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.10.py"
BACKUP_FILE = CHATBOT_FILE + ".bak"

# New generate_response function
NEW_FUNCTION = """
def generate_response(prompt, system_prompt=None):
    """Generate a response from Ollama API"""
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
        logger.info(f"Request data: {data}")
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
"""

def update_chatbot():
    """Update the generate_response function in the chatbot file"""
    print(f"Updating chatbot file: {CHATBOT_FILE}")
    
    # Create backup
    print(f"Creating backup: {BACKUP_FILE}")
    shutil.copy2(CHATBOT_FILE, BACKUP_FILE)
    
    # Read the file
    with open(CHATBOT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the generate_response function
    pattern = r'def generate_response\([^)]*\):.*?(?=\n# Process user question|\n\n# Process)'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        # Replace the function
        print("Found generate_response function, replacing it...")
        new_content = content.replace(match.group(0), NEW_FUNCTION.strip())
        
        # Write the updated content
        with open(CHATBOT_FILE, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print("Chatbot updated successfully!")
        print(f"Original file backed up to: {BACKUP_FILE}")
        print(f"Updated file: {CHATBOT_FILE}")
        return True
    else:
        print("Could not find generate_response function in the chatbot file.")
        return False

if __name__ == "__main__":
    update_chatbot()
