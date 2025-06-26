import requests

OLLAMA_HOST = "http://10.10.12.202:11434"

# Test connection and list models
def test_ollama_api():
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags")
        resp.raise_for_status()
        data = resp.json()
        models = data.get("models", [])
        print("Ollama API is reachable!")
        print("Available models:")
        for model in models:
            print("-", model.get("name"))
    except Exception as e:
        print("Could not connect to Ollama API:", e)

if __name__ == "__main__":
    test_ollama_api()
