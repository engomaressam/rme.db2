import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Simple test
print('Testing OpenAI connection...')
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": "Say hi"
    }],
    temperature=0.1,
    max_tokens=10
)

print(f"Response: {response.choices[0].message.content}")
