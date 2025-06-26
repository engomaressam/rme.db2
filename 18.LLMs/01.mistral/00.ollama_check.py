import requests

def check_ollama_running():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            print("Ollama is running and ready!")
            print("Available models:", response.json().get("models", []))
            return True
        else:
            print(f"Ollama responded with status code: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("Ollama is not running. Please start the Ollama server.")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

if __name__ == "__main__":
    check_ollama_running() 