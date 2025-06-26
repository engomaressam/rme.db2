"""
Simple Alstom Project Assistant Chatbot
This is a simplified version that focuses on connecting to the Ollama API
"""

import os
import time
import logging
import requests
import gradio as gr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simple_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SimpleChat")

# Ollama API configuration
OLLAMA_API_URL = "http://10.10.12.202:11434"
MODEL_NAME = "qwen2.5-coder:7b"

def check_ollama_availability():
    """Check if Ollama API is available"""
    try:
        logger.info(f"Checking Ollama API at {OLLAMA_API_URL}")
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            logger.info(f"Available models: {', '.join(model_names)}")
            return True, model_names
        else:
            logger.error(f"Error connecting to Ollama API: {response.status_code}")
            return False, []
    except Exception as e:
        logger.exception(f"Exception checking Ollama API: {e}")
        return False, []

def generate_response(prompt, system_prompt=None):
    """Generate a response from Ollama API"""
    try:
        logger.info(f"Generating response using {MODEL_NAME}")
        
        # Prepare request data
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            data["system"] = system_prompt
        
        # Send request
        logger.info("Sending request to Ollama API...")
        start_time = time.time()
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate", 
            json=data,
            timeout=60
        )
        end_time = time.time()
        
        # Process response
        logger.info(f"Response time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', 'No response')
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        logger.exception(error_msg)
        return error_msg

def respond(message, history):
    """Process user message and generate response"""
    if not message:
        return "", history
    
    # Simple system prompt
    system_prompt = "You are the Alstom Project Assistant, a helpful AI that provides information about the Alstom project."
    
    # Generate response
    response = generate_response(message, system_prompt)
    
    # Update history
    history.append((message, response))
    return "", history

def main():
    """Main function to run the chatbot"""
    # Check Ollama API
    api_available, available_models = check_ollama_availability()
    
    if not api_available:
        logger.error("Ollama API is not available. Please check the connection.")
        status_html = "<span style='color: red'>❌ Ollama API not available</span>"
    else:
        logger.info("Ollama API is available")
        status_html = "<span style='color: green'>✓ Connected to Ollama API</span>"
    
    # Create Gradio interface
    with gr.Blocks(title="Alstom Project Assistant") as demo:
        gr.Markdown("# Alstom Project Assistant")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Sidebar
                gr.Markdown("## Status")
                gr.HTML(status_html)
                gr.Markdown(f"API URL: {OLLAMA_API_URL}")
                gr.Markdown(f"Model: {MODEL_NAME}")
                
                if available_models:
                    gr.Markdown("### Available Models")
                    for model in available_models:
                        gr.Markdown(f"- {model}")
            
            with gr.Column(scale=3):
                # Chat interface
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(
                    placeholder="Ask a question about the Alstom project...",
                    show_label=False
                )
                
                # Set up event handlers
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
        
        # Launch the interface
        demo.launch(server_name="0.0.0.0", share=False)

if __name__ == "__main__":
    main()
