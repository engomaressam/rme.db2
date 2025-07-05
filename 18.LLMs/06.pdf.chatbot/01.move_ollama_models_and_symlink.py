import os
import shutil
from pathlib import Path
import subprocess

# Hardcoded paths
OLLAMA_MODELS_C = Path(r"C:/Users/Omar Essam2/.ollama/models")
OLLAMA_MODELS_D = Path(r"D:/OEssam/models")

print(f"Ollama models folder on C: {OLLAMA_MODELS_C}")
print(f"Target folder on D: {OLLAMA_MODELS_D}")

# 1. Move models if needed
if OLLAMA_MODELS_C.exists() and any(OLLAMA_MODELS_C.iterdir()):
    OLLAMA_MODELS_D.mkdir(parents=True, exist_ok=True)
    print("Moving existing models to D drive...")
    for item in OLLAMA_MODELS_C.iterdir():
        dest = OLLAMA_MODELS_D / item.name
        if dest.exists():
            print(f"Already exists: {dest}, skipping.")
        else:
            shutil.move(str(item), str(dest))
    print("Move complete.")
else:
    print("No models found to move or already moved.")

# 2. Remove old models folder
if OLLAMA_MODELS_C.exists():
    try:
        shutil.rmtree(OLLAMA_MODELS_C)
        print("Old C drive models folder deleted.")
    except Exception as e:
        print(f"Could not delete old models folder: {e}")

# 3. Create symlink (will likely fail unless run as admin)
OLLAMA_MODELS_C.parent.mkdir(parents=True, exist_ok=True)
if not OLLAMA_MODELS_C.exists():
    try:
        subprocess.check_call(["mklink", "/D", str(OLLAMA_MODELS_C), str(OLLAMA_MODELS_D)], shell=True)
        print(f"Symlink created: {OLLAMA_MODELS_C} -> {OLLAMA_MODELS_D}")
    except Exception as e:
        print(f"Failed to create symlink: {e}")
else:
    print("Symlink or folder already exists at models location.")

# 4. Test access
print("\nTesting access via symlink...")
try:
    files = list(OLLAMA_MODELS_C.glob("**/*"))
    print(f"Found {len(files)} files via symlink. Sample:")
    for f in files[:10]:
        print(f"  {f}")
    print("Symlink test successful.")
except Exception as e:
    print(f"Symlink test failed: {e}")
