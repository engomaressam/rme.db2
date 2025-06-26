import requests
import sys
import time

# URLs to check
urls = [
    "http://10.10.12.202:11434",  # Remote GPU machine
    "http://localhost:11434",     # Local machine
    "http://127.0.0.1:11434"      # Localhost IP
]

def check_url(url):
    print(f"\nChecking Ollama API at {url}...")
    try:
        # Check if API is reachable
        response = requests.get(f"{url}/api/tags", timeout=10)
        if response.status_code == 200:
            print(f"✓ API is reachable at {url}")
            
            # Get available models
            models = response.json().get("models", [])
            if models:
                print(f"✓ Found {len(models)} models:")
                for model in models:
                    print(f"  - {model.get('name')}")
                
                # Test the first model with a simple prompt
                test_model = models[0].get('name')
                print(f"\nTesting model {test_model} with a simple prompt...")
                try:
                    start_time = time.time()
                    test_response = requests.post(
                        f"{url}/api/generate",
                        json={
                            "model": test_model,
                            "prompt": "Hello, are you working?",
                            "stream": False
                        },
                        timeout=30
                    )
                    elapsed_time = time.time() - start_time
                    
                    if test_response.status_code == 200:
                        response_text = test_response.json().get("response", "")
                        print(f"✓ Model responded in {elapsed_time:.2f} seconds:")
                        print(f"  Response: {response_text[:100]}...")
                        return True
                    else:
                        print(f"✗ Model test failed with status code {test_response.status_code}")
                except Exception as e:
                    print(f"✗ Error testing model: {str(e)}")
            else:
                print("✗ No models found")
        else:
            print(f"✗ API returned status code {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"✗ Connection timed out")
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error - API not reachable")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
    
    return False

def main():
    print("OLLAMA API STATUS CHECK")
    print("======================")
    
    success = False
    for url in urls:
        if check_url(url):
            success = True
            print(f"\n✓ Successfully connected to Ollama API at {url}")
            break
    
    if not success:
        print("\n✗ Failed to connect to Ollama API at any URL")
        sys.exit(1)
    else:
        print("\nOllama API is working correctly!")

if __name__ == "__main__":
    main()
