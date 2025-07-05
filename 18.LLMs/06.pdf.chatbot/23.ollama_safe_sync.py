import os
import shutil

def sync_dir(src, dst):
    if not os.path.exists(src):
        print(f'Source folder does not exist: {src}')
        return
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        dst_root = os.path.join(dst, rel_root) if rel_root != '.' else dst
        os.makedirs(dst_root, exist_ok=True)
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_root, file)
            # Skip partial files
            if '-partial' in file:
                continue
            if not os.path.exists(dst_file):
                print(f'Copying {src_file} -> {dst_file}')
                try:
                    shutil.copy2(src_file, dst_file)
                except Exception as e:
                    print(f'Failed to copy {src_file}: {e}')
            else:
                # Uncomment next line for verbose mode
                # print(f'Skipping existing: {dst_file}')
                pass

if __name__ == '__main__':
    network_base = r'H:\Projects Control (PC)\10 Backup\07 Ollama\models'
    local_base = r'D:\OEssam\models'
    # Sync blobs and manifests
    for folder in ['blobs', 'manifests']:
        src = os.path.join(network_base, folder)
        dst = os.path.join(local_base, folder)
        print(f'\n=== Syncing {folder} ===')
        sync_dir(src, dst)
    print('\nSync complete. All missing models from network are now on D: without deleting any existing models.')
