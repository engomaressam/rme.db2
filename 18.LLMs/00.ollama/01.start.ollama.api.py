import os
import subprocess
import time

def set_ollama_env():
    """Set optimal environment variables for Ollama"""
    env_vars = {
        'OLLAMA_HOST': '0.0.0.0:11434',  # Listen on all interfaces
        'OLLAMA_CUDA': '1',              # Enable CUDA
        'OLLAMA_GPU_LAYERS': '99',       # Use maximum GPU layers
        'OLLAMA_FLASH_ATTENTION': 'true', # Enable Flash Attention
        'OLLAMA_GPU_MEMORY': '22'        # Allocate 22GB of GPU memory (for 24GB GPU)
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")



def start_ollama_server():
    """Start the Ollama server with optimized settings"""
    # Set environment variables
    set_ollama_env()
    
    try:
        # Kill any existing Ollama processes
        print("Checking for existing Ollama processes...")
        subprocess.run(['taskkill', '/F', '/IM', 'ollama.exe'], capture_output=True)
        time.sleep(3)  # Wait longer to ensure process is fully terminated
        
        # Start Ollama server
        print("\nStarting Ollama server...")
        process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(5)  # Wait longer for full initialization
        
        print("\nOllama server started successfully!")
        print("API URL: http://0.0.0.0:11434")
        print("\nAvailable Models:")
        subprocess.run(['ollama', 'list'], check=True)
        
        print("\n" + "=" * 50)
        print("Server Status Dashboard")
        print("=" * 50)
        print("1. Keep this window open to maintain the API connection")
        print("2. The API is accessible from other machines at:")
        print("   http://10.10.12.202:11434")
        print("3. Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Keep the script running
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print("\nOllama server stopped unexpectedly!")
                break
                
    except KeyboardInterrupt:
        print("\nShutting down Ollama server...")
        process.terminate()
        process.wait()
        print("Server stopped.")
    except Exception as e:
        print(f"Error starting Ollama server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Starting Ollama API Server with GPU Optimization")
    print("=" * 50)
    
    # Check if ollama is installed
    try:
        subprocess.run(['ollama', '--version'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: Ollama is not installed or not in PATH")
        input("Press Enter to exit...")
        exit(1)
    
    try:
        # Start the server
        start_ollama_server()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\n\nError: {e}")
    
    print("\nServer stopped.")
    input("Press Enter to exit...")

