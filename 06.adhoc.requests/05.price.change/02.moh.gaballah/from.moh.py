import pandas as pd
import glob
import os
from tqdm import tqdm

def list_now():
    print(f"Current working directory: {os.getcwd()}")  # Print the current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change the working directory to the script's directory
    print(f"New working directory: {os.getcwd()}")  # Confirm the new working directory

    # Step 1: Find and read the only xlsx file in the directory
    current_dir = os.getcwd()
    file_list = os.listdir(current_dir)
    print(f"Files found: {file_list}")  # Debug print statement
    results = []
    for file in tqdm(os.listdir(current_dir)):
        if file.endswith('.xlsb'):
            file_path = os.path.join(current_dir, file)
            df = pd.read_excel(file_path, engine='pyxlsb')
            # Step 2: Extract relevant columns including the additional ones
            df = df[['description', 'unit', 'unit_price', 'approved_date', 'project_name', 'vendor', 'qty', 'amount_egp', 'project_no', 'organization_code', 'buyer_dept', 'buyer', 'qty_received']]

            # Step 3: Convert 'approved_date' to datetime and determine the week number
            df['approved_date'] = pd.to_datetime(df['approved_date'])
            df['week'] = df['approved_date'].dt.isocalendar().week
            df['Year'] = file.split('.')[0]

            # Step 4: Group by 'description', 'unit', and 'week'
            grouped = df.groupby(['description', 'unit', 'week'])

            # Step 5 & 6: Check for variations in 'unit_price' within each group
            for name, group in grouped:
                max_price = group['unit_price'].max()
                min_price = group['unit_price'].min()
                if max_price > min_price * 1.05:
                    results.append(group)

    # Combine all results into a single DataFrame
    if results:
        result_df = pd.concat(results)
    else:
        result_df = pd.DataFrame()

    # Output the result
    result_df.to_excel('output.xlsx', index=False)

    print("Analysis complete. Results are saved in 'output.xlsx'.")

# Test the module independently
if __name__ == "__main__":
    list_now()
