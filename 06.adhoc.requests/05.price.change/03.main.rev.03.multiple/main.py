import same_week
import two_weeks
import os

def main():
    print(f"Current working directory: {os.getcwd()}")  # Print the current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change the working directory to the script's directory
    print(f"New working directory: {os.getcwd()}")  # Confirm the new working directory
    
    same_week.list_now()

if __name__ == "__main__":
    main()
