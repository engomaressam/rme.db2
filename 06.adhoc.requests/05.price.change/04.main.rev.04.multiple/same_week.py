import pandas as pd
import glob
import os

def list_now():
    print(f"Current working directory: {os.getcwd()}")  # Print the current working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change the working directory to the script's directory
    print(f"New working directory: {os.getcwd()}")  # Confirm the new working directory

    # Step 1: Find and read all xlsx files with "PO_Follow_up" in the name
    file_list = glob.glob('*PO_Follow_up*.xlsx')
    print(f"Files found: {file_list}")  # Debug print statement

    if not file_list:
        raise ValueError("No .xlsx files with 'PO_Follow_up' in the name found in the directory.")

    # Initialize an empty DataFrame to store results
    all_results = pd.DataFrame()

    for file_path in file_list:
        df = pd.read_excel(file_path)

        # Step 2: Extract relevant columns including the additional ones
        df = df[['project_no', 'project_name', 'organization_code', 'po_no', 'pr_no', 'pr_category', 'po_status',
                 'shipment_cancel_status', 'shipment_close_status', 'next_approver', 'vendor', 'vendor_no', 'buyer',
                 'buyer_dept', 'po_line', 'creation_date', 'approved_date', 'pr_line', 'pr_reason', 'po_comments',
                 'store_code', 'description', 'qty', 'qty_cancelled', 'unit', 'unit_price', 'unit_price_without_tax',
                 'currency', 'amount', 'amount_egp', 'amount_without_tax', 'amount_egp_without_tax', 'tax_amount_egp',
                 'tax_amount', 'tax_code', 'task', 'task_name', 'expenditure_type', 'expenditure_category', 'term',
                 'qty_received', 'qty_accepted', 'qty_rejected', 'qty_delivered', 'qty_open', 'qty_open_amount', 'docs']]

        # Step 3: Convert 'approved_date' to datetime and determine the week number
        df['approved_date'] = pd.to_datetime(df['approved_date'])
        df['week'] = df['approved_date'].dt.isocalendar().week

        # Step 4: Group by 'description', 'unit', and 'week'
        grouped = df.groupby(['description', 'unit', 'week'])

        # Step 5 & 6: Check for variations in 'unit_price' within each group
        results = []

        for name, group in grouped:
            max_price = group['unit_price'].max()
            min_price = group['unit_price'].min()
            if max_price > min_price * 1.05:
                results.append(group)

        # Combine all results into a single DataFrame for the current file
        if results:
            result_df = pd.concat(results)
            all_results = pd.concat([all_results, result_df])

    # Output the result
    if not all_results.empty:
        all_results.to_excel('same_week_list.xlsx', index=False)

    print("Analysis complete. Results are saved in 'output.xlsx'.")

# Test the module independently
if __name__ == "__main__":
    list_now()
