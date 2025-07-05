import os

NETWORK_BLOBS = r'H:\Projects Control (PC)\10 Backup\07 Ollama\models\blobs'
LOCAL_BLOBS = r'D:\OEssam\models\blobs'

# Collect all blob files in both dirs
network_blobs = {f: os.path.getsize(os.path.join(NETWORK_BLOBS, f)) for f in os.listdir(NETWORK_BLOBS) if os.path.isfile(os.path.join(NETWORK_BLOBS, f))}
local_blobs = {f: os.path.getsize(os.path.join(LOCAL_BLOBS, f)) for f in os.listdir(LOCAL_BLOBS) if os.path.isfile(os.path.join(LOCAL_BLOBS, f))}

flagged = []
for fname, net_size in network_blobs.items():
    local_size = local_blobs.get(fname)
    if local_size is not None and local_size < net_size:
        flagged.append((fname, local_size, net_size))

# Always delete local partial files
partial_files = [f for f in os.listdir(LOCAL_BLOBS) if '-partial' in f]
if partial_files:
    print('\nDeleting local partial files:')
    for f in partial_files:
        try:
            os.remove(os.path.join(LOCAL_BLOBS, f))
            print(f'  Deleted {f}')
        except Exception as e:
            print(f'  Failed to delete {f}: {e}')
else:
    print('No local partial files found.')

if flagged:
    print('\nThe following local blob files are smaller than the network version (likely incomplete/corrupted):')
    for fname, lsize, nsize in flagged:
        print(f'  {fname}: local={lsize} bytes, network={nsize} bytes')
    print('\nYou can delete these files and re-run the sync script to fix them.')
    # Optional: Delete them automatically
    delete = input('Delete these incomplete local files now? (y/N): ').strip().lower()
    if delete == 'y':
        for fname, _, _ in flagged:
            try:
                os.remove(os.path.join(LOCAL_BLOBS, fname))
                print(f'Deleted {fname}')
            except Exception as e:
                print(f'Failed to delete {fname}: {e}')
else:
    print('No incomplete/corrupted local blobs found. All local blobs are at least as large as the network versions.')
