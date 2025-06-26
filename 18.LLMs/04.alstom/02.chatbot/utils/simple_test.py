"""
Simple test for Ollama API
"""
import requests
import time

# Ollama API URL
OLLAMA_API_URL = "http://10.10.12.202:11434"
MODEL_NAME = "qwen2.5-coder:7b"

print(f"Testing Ollama API at {OLLAMA_API_URL} with model {MODEL_NAME}")

# Prepare request data
data = {
    "model": MODEL_NAME,
    "prompt": "What is the capital of France?",
    "stream": False
}

# Send request
print("Sending request...")
start_time = time.time()
try:
    response = requests.post(
        f"{OLLAMA_API_URL}/api/generate", 
        json=data,
        timeout=30
    )
    end_time = time.time()
    
    # Process response
    print(f"Response time: {end_time - start_time:.2f} seconds")
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nResponse:")
        print(result.get('response', 'No response'))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Exception: {e}")
