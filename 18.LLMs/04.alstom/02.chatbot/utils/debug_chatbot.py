"""
Debug version of the Alstom chatbot
This script tests the specific generate_response function from the chatbot
"""

import requests
import json
import time
import gradio as gr

# Ollama API configuration
OLLAMA_API_URL = "http://10.10.12.202:11434"
MODEL_NAME = "qwen2.5-coder:7b"

def generate_response(prompt, system_prompt=None):
    """Generate a response from Ollama API"""
    try:
        print(f"Generating response using {MODEL_NAME}...")
        print(f"API URL: {OLLAMA_API_URL}")
        
        # Prepare request data
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            data["system"] = system_prompt
            print(f"Using system prompt: {system_prompt}")
        
        # Send request with longer timeout
        print("Sending request to Ollama API...")
        start_time = time.time()
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate", 
            json=data,
            timeout=60  # Longer timeout
        )
        end_time = time.time()
        
        # Process response
        if response.status_code == 200:
            result = response.json()
            print(f"Response received in {end_time - start_time:.2f} seconds")
            return result.get('response', 'No response')
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        print(f"Exception in generate_response: {e}")
        return f"Error: {str(e)}"

def simple_chat_interface():
    """Create a simple chat interface for testing"""
    with gr.Blocks() as demo:
        gr.Markdown("# Alstom Chatbot Debug Interface")
        
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(placeholder="Type a message...")
        
        def respond(message, history):
            # Simple response generation
            system_prompt = "You are an assistant for the Alstom project."
            bot_message = generate_response(message, system_prompt)
            history.append((message, bot_message))
            return "", history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        
    demo.launch(share=False)

if __name__ == "__main__":
    print(f"Testing with Ollama API at {OLLAMA_API_URL}")
    
    # Test direct response generation
    test_response = generate_response(
        "What is the capital of France?",
        "You are a helpful assistant."
    )
    print("\nTest response:")
    print(test_response)
    
    # Launch simple chat interface
    print("\nStarting simple chat interface...")
    simple_chat_interface()
