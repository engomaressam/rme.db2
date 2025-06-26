import subprocess
import time

MODELS = [
    "llama3:70b",
    "command-r-plus"
]
RETRY_INTERVAL = 10 * 60  # 10 minutes in seconds


def pull_model(model_name):
    while True:
        print(f"\n[INFO] Attempting to pull model: {model_name}")
        try:
            result = subprocess.run([
                "ollama", "pull", model_name
            ], check=True)
            print(f"[SUCCESS] Successfully pulled {model_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to pull {model_name}. Will retry in {RETRY_INTERVAL // 60} minutes.")
            time.sleep(RETRY_INTERVAL)
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}. Will retry in {RETRY_INTERVAL // 60} minutes.")
            time.sleep(RETRY_INTERVAL)


def main():
    for model in MODELS:
        pull_model(model)
    print("\n[INFO] All models pulled successfully!")


if __name__ == "__main__":
    main() 