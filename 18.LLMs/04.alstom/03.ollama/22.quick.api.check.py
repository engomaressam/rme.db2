"""
Quick Ollama API Check
This script performs a quick check of the Ollama API with a short timeout.
"""

import requests
import sys
import time

def check_api(url, timeout=3):
    print(f"Checking Ollama API at {url} with timeout {timeout}s...")
    try:
        start_time = time.time()
        response = requests.get(f"{url}/api/tags", timeout=timeout)
        elapsed = time.time() - start_time
        
        print(f"Response received in {elapsed:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            print(f"Success! Found {len(models)} models:")
            for model in models:
                print(f"  - {model.get('name')}")
            return True
        else:
            print(f"API returned error status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"Error: Connection timed out after {timeout} seconds")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"Error: Connection failed - {str(e)}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    api_url = "http://10.10.12.202:11434"
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    
    success = check_api(api_url)
    if not success:
        sys.exit(1)
