import requests
import json

try:
    response = requests.get('http://10.10.12.202:11434/api/tags')
    data = response.json()
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Error: {str(e)}")
