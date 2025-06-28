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
            print(f"[DEBUG] Executing command: {PULL_COMMAND}")
            result = subprocess.run(
                PULL_COMMAND,
                shell=True,
                text=True,
                capture_output=True
            )
            # Print the output from the command
            if result.stdout:
                print("Command output:")
                print(result.stdout)
            if result.stderr:
                print("Command error output:")
                print(result.stderr)
                
            # Check the return code manually to handle it better
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    returncode=result.returncode,
                    cmd=PULL_COMMAND,
                    output=result.stdout,
                    stderr=result.stderr
                )
                
            print(f"[SUCCESS] Successfully pulled {model_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Command failed with return code {e.returncode}")
            if e.stdout:
                print("Standard Output:")
                print(e.stdout)
            if e.stderr:
                print("Error Output:")
                print(e.stderr)
                
            print(f"[ERROR] Failed to pull {model_name}.")
            if attempt < MAX_RETRIES:
                wait_minutes = RETRY_INTERVAL // 60
                print(f"[INFO] Will retry in {wait_minutes} minutes... (Attempt {attempt + 1}/{MAX_RETRIES})")
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
