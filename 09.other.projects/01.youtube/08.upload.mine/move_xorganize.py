import os
import shutil


def get_input_directory(prompt):
    while True:
        path = input(prompt).strip('"')
        if os.path.isdir(path):
            return os.path.abspath(path)
        else:
            print(f"Directory '{path}' does not exist. Please try again.")

def main():
    print("This script will move all 'xorganize' folders from a source directory to a destination directory, preserving their relative paths.")
    src_root = get_input_directory("Enter the source (input directory 1): ")
    dst_root = get_input_directory("Enter the destination (input directory 2): ")

    for dirpath, dirnames, filenames in os.walk(src_root, topdown=True):
        # Make a copy of dirnames so we can modify it while iterating
        for dirname in list(dirnames):
            if dirname == "xorganize":
                src_xorganize = os.path.join(dirpath, dirname)
                # Compute relative path from src_root
                rel_path = os.path.relpath(dirpath, src_root)
                # Target parent directory in dst_root
                dst_parent = os.path.join(dst_root, rel_path)
                dst_xorganize = os.path.join(dst_parent, "xorganize")

                # Ensure the parent directories exist
                os.makedirs(dst_parent, exist_ok=True)

                print(f"Moving '{src_xorganize}' to '{dst_xorganize}'")
                shutil.move(src_xorganize, dst_xorganize)
                # Remove 'xorganize' from dirnames so os.walk doesn't descend into it
                dirnames.remove(dirname)

if __name__ == "__main__":
    main()
