import subprocess
import time

MODEL = "volker-mauel/Kimi-Dev-72B-GGUF"
PULL_COMMAND = f"ollama pull {MODEL}"
RETRY_INTERVAL = 10 * 60  # 10 minutes in seconds
MAX_RETRIES = 10  # Maximum number of retry attempts

def pull_model(model_name):
    attempt = 1
    while attempt <= MAX_RETRIES:
        print(f"\n[INFO] Attempt {attempt}/{MAX_RETRIES} - Pulling model: {model_name}")
        try:
            # Using shell=True to execute the exact command as provided
            result = subprocess.run(
                PULL_COMMAND,
                shell=True,
                check=True,
                text=True,
                capture_output=True
            )
            # Print the output from the command
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"[INFO] {result.stderr}")
                
            print(f"[SUCCESS] Successfully pulled {model_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to pull {model_name}. Error: {e}")
            if attempt < MAX_RETRIES:
                print(f"[INFO] Will retry in {RETRY_INTERVAL // 60} minutes...")
                time.sleep(RETRY_INTERVAL)
            attempt += 1
            
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            if attempt < MAX_RETRIES:
                print(f"[INFO] Will retry in {RETRY_INTERVAL // 60} minutes...")
                time.sleep(RETRY_INTERVAL)
            attempt += 1
    
    print(f"[ERROR] Failed to pull {model_name} after {MAX_RETRIES} attempts.")
    return False

def main():
    print(f"[INFO] Starting to pull model using command: {PULL_COMMAND}")
    success = pull_model(MODEL)
    
    if success:
        print("\n[SUCCESS] Model pulled successfully!")
    else:
        print("\n[ERROR] Failed to pull the model after multiple attempts.")
        print("Please check your internet connection and Ollama installation, then try again.")

if __name__ == "__main__":
    main()
