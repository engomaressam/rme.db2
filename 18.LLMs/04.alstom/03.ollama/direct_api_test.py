"""
Direct test of Ollama API
This script tests the Ollama API directly with a simple completion request
"""

import requests
import json
import time

# Ollama API URL
OLLAMA_API_URL = "http://10.10.12.202:11434"

def test_models():
    """Check available models"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print("Available models:")
            for model in models.get('models', []):
                print(f"- {model['name']}")
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_completion(model_name="qwen2.5-coder:7b"):
    """Test a simple completion"""
    try:
        print(f"\nTesting completion with model: {model_name}")
        
        # Request data
        data = {
            "model": model_name,
            "prompt": "What is the capital of France?",
            "stream": False
        }
        
        # Send request
        print("Sending request...")
        start_time = time.time()
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate", 
            json=data,
            timeout=30
        )
        end_time = time.time()
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print("\nResponse:")
            print(result.get('response', 'No response'))
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    print(f"Testing Ollama API at {OLLAMA_API_URL}")
    
    # Test models
    if test_models():
        # If models are available, test completion with each model
        test_completion("qwen2.5-coder:7b")
        # Uncomment to test other models
        # test_completion("llama4:17b-scout-16e-instruct-q4_K_M")
        # test_completion("deepseek-coder:6.7b")
        # test_completion("llama3.2:latest")
    else:
        print("Could not retrieve models. API may be unavailable.")
