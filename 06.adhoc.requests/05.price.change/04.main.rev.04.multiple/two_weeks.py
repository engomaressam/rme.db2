import pandas as pd
import glob
import os

def list_now():
    print(f"Current working directory: {os.getcwd()}")  # Print the current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change the working directory to the script's directory
    print(f"New working directory: {os.getcwd()}")  # Confirm the new working directory

    # Step 1: Find and read all xlsx files in the directory with "PO_Follow_up" in the name
    file_list = glob.glob('*PO_Follow_up*.xlsx')
    print(f"Files found: {file_list}")  # Debug print statement
    if not file_list:
        raise ValueError("No Excel files with 'PO_Follow_up' in the name found in the directory.")
    
    results = []

    for file_path in file_list:
        df = pd.read_excel(file_path)

        # Step 2: Extract relevant columns including the additional ones
        df = df[['description', 'unit', 'unit_price', 'approved_date', 'project_name', 'vendor', 'qty', 'amount_egp', 'project_no', 'organization_code', 'buyer_dept', 'buyer', 'qty_received']]

        # Step 3: Convert 'approved_date' to datetime and determine the week number
        df['approved_date'] = pd.to_datetime(df['approved_date'])
        df['week'] = df['approved_date'].dt.isocalendar().week
        df['year'] = df['approved_date'].dt.year  # Use dt.year instead of dt.isocalendar().year

        # Step 4: Group by 'description', 'unit', 'year', and 'week'
        grouped = df.groupby(['description', 'unit', 'year', 'week'])

        # Step 5 & 6: Check for variations in 'unit_price' between consecutive weeks
        for name, group in grouped:
            description, unit, year, week = name
            # Check if there is a group for the next week
            next_week_group = grouped.get_group((description, unit, year, week + 1)) if (description, unit, year, week + 1) in grouped.groups else None
            if next_week_group is not None:
                max_price_current = group['unit_price'].max()
                min_price_next = next_week_group['unit_price'].min()
                if min_price_next > max_price_current * 1.05:
                    results.append(group)
                    results.append(next_week_group)
            
            # Handle year-end transition
            next_year_group = grouped.get_group((description, unit, year + 1, 1)) if (description, unit, year + 1, 1) in grouped.groups else None
            if next_year_group is not None and week == 52:
                max_price_current = group['unit_price'].max()
                min_price_next = next_year_group['unit_price'].min()
                if min_price_next > max_price_current * 1.05:
                    results.append(group)
                    results.append(next_year_group)

    # Combine all results into a single DataFrame
    if results:
        result_df = pd.concat(results).drop_duplicates()
    else:
        result_df = pd.DataFrame(columns=[
            'description', 'unit', 'unit_price', 'approved_date', 'project_name', 'vendor', 
            'qty', 'amount_egp', 'project_no', 'organization_code', 'buyer_dept', 'buyer', 'qty_received', 'week', 'year'
        ])

    # Output the result
    result_df.to_excel('two_weeks_list.xlsx', index=False)

    print("Analysis complete. Results are saved in 'output.xlsx'.")

# Test the module independently
if __name__ == "__main__":
    list_now()
