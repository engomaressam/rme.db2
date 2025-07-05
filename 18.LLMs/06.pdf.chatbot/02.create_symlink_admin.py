import subprocess

OLLAMA_MODELS_C = r"C:\Users\Omar Essam2\.ollama\models"
OLLAMA_MODELS_D = r"D:\OEssam\models"

try:
    subprocess.check_call([
        "cmd", "/c", f'mklink /D "{OLLAMA_MODELS_C}" "{OLLAMA_MODELS_D}"'
    ], shell=True)
    print(f"Symlink created: {OLLAMA_MODELS_C} -> {OLLAMA_MODELS_D}")
except Exception as e:
    print(f"Failed to create symlink: {e}")
