"""
Fix for the Alstom chatbot
"""
import os
import shutil
import re

# Path to the chatbot file
CHATBOT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.10.py"
BACKUP_FILE = CHATBOT_FILE + ".bak"
OUTPUT_FILE = "c:\\Users\\Omar Essam2\\OneDrive - Rowad Modern Engineering\\x004 Data Science\\03.rme.db\\00.repo\\rme.db\\18.LLMs\\04.alstom\\15.vector_chatbot.rev.11.py"

# Create backup
print(f"Creating backup: {BACKUP_FILE}")
shutil.copy2(CHATBOT_FILE, BACKUP_FILE)

# Read the file
with open(CHATBOT_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the generate_response function
generate_response_pattern = r'def generate_response\(prompt, system_prompt=None\):.*?(?=def process_question|# Process user question)'
generate_response_replacement = '''def generate_response(prompt, system_prompt=None):
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
        
        # Log the request data
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

'''

# Replace the function
new_content = re.sub(generate_response_pattern, generate_response_replacement, content, flags=re.DOTALL)

# Write the updated content
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Original file backed up to: {BACKUP_FILE}")
print(f"Fixed file written to: {OUTPUT_FILE}")
print("\nTo use the fixed chatbot, run:")
print(f"python {OUTPUT_FILE}")
