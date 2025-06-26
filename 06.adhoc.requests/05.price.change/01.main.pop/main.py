import same_week
import two_weeks
import os

def main():
    print(f"Current working directory: {os.getcwd()}")  # Print the current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change the working directory to the script's directory
    print(f"New working directory: {os.getcwd()}")  # Confirm the new working directory

    while True:
        print("\nSelect an option to run:")
        print("1. Run same_week")
        print("2. Run two_weeks")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            same_week.list_now()
        elif choice == '2':
            two_weeks.list_now()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()