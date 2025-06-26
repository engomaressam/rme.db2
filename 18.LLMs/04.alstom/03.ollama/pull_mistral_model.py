import requests
import time
import sys

# Ollama API URL
OLLAMA_API_URL = "http://10.10.12.202:11434"

def check_model_exists(model_name):
    """Check if the model already exists on the server"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get('name', '') for model in models]
            print(f"Available models: {model_names}")
            
            # Check if any model contains our model name
            for model in model_names:
                if model_name in model:
                    print(f"✓ Model {model_name} already exists as {model}")
                    return True
            
            print(f"✗ Model {model_name} not found")
            return False
        else:
            print(f"✗ API returned status code {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error checking model: {str(e)}")
        return False

def pull_model(model_name):
    """Pull the model from Ollama API"""
    print(f"Starting to pull model {model_name}...")
    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=None  # No timeout for pulling models
        )
        
        if response.status_code == 200:
            # Process the streaming response to show progress
            total_size = 0
            start_time = time.time()
            last_update_time = start_time
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = line.decode('utf-8')
                        # Print progress data
                        current_time = time.time()
                        if current_time - last_update_time >= 2:  # Update every 2 seconds
                            elapsed = current_time - start_time
                            print(f"Still pulling... {elapsed:.1f} seconds elapsed. {data}")
                            last_update_time = current_time
                    except Exception as e:
                        print(f"Error parsing response: {str(e)}")
            
            elapsed_time = time.time() - start_time
            print(f"✓ Successfully pulled model {model_name} in {elapsed_time:.1f} seconds")
            return True
        else:
            print(f"✗ Failed to pull model: Status code {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error pulling model: {str(e)}")
        return False

def test_model(model_name):
    """Test the model with a simple prompt"""
    print(f"Testing model {model_name}...")
    try:
        start_time = time.time()
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello, are you working?",
                "stream": False
            },
            timeout=30
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            response_text = response.json().get("response", "")
            print(f"✓ Model responded in {elapsed_time:.2f} seconds:")
            print(f"  Response: {response_text[:100]}...")
            return True
        else:
            print(f"✗ Model test failed with status code {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing model: {str(e)}")
        return False

def main():
    model_name = "mistral:latest"
    
    print("MISTRAL MODEL PULLER")
    print("===================")
    print(f"Ollama API URL: {OLLAMA_API_URL}")
    
    # Check if model already exists
    if check_model_exists(model_name):
        print("Model already exists, testing it...")
        if test_model(model_name):
            print("✓ Model is working correctly!")
            return
        else:
            print("Model exists but test failed. Will try to pull it again.")
    
    # Pull the model
    if pull_model(model_name):
        # Test the model after pulling
        if test_model(model_name):
            print("✓ Model is working correctly!")
        else:
            print("✗ Model was pulled but test failed.")
    else:
        print("✗ Failed to pull the model.")
        sys.exit(1)

if __name__ == "__main__":
    main()
