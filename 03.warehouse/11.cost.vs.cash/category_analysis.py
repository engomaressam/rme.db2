from db_connection import DatabaseConnection
import pandas as pd
import os

def analyze_by_category(cost_data, cash_data, group_by_column):
    """
    Analyze cost and cash data grouped by a specific column
    Returns both summary and detailed DataFrames
    """
    # Get cost totals by category
    cost_by_category = cost_data.groupby([group_by_column, 'VENDOR_NAME'])['AMOUNT'].sum().reset_index()
    
    # Get total cash per vendor (since cash data might not have the category)
    cash_by_vendor = cash_data.groupby('Supplier Name')['AMOUNT'].sum()
    
    # Add cash amounts to the cost data
    cost_by_category['CASH_AMOUNT'] = cost_by_category['VENDOR_NAME'].map(cash_by_vendor).fillna(0)
    
    # Create summary by category
    summary = cost_by_category.groupby(group_by_column).agg({
        'AMOUNT': 'sum',
        'CASH_AMOUNT': 'sum',
        'VENDOR_NAME': 'count'
    })
    summary.columns = ['Total_Cost', 'Total_Cash', 'Vendor_Count']
    # Convert to integers
    summary['Total_Cost'] = summary['Total_Cost'].fillna(0).astype('int64')
    summary['Total_Cash'] = summary['Total_Cash'].fillna(0).astype('int64')
    summary = summary.sort_values('Total_Cost', ascending=False)
    
    # Create detailed view with vendors per category
    detailed = cost_by_category.sort_values(['AMOUNT'], ascending=[False])
    detailed['AMOUNT'] = detailed['AMOUNT'].fillna(0).astype('int64')
    detailed['CASH_AMOUNT'] = detailed['CASH_AMOUNT'].fillna(0).astype('int64')
    
    return summary, detailed

def export_to_excel(df, sheet_name, writer, start_col=0):
    """Export DataFrame to Excel with proper formatting"""
    # Convert numeric columns to integers before exporting
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0).astype('int64')
    
    df.to_excel(writer, sheet_name=sheet_name, startcol=start_col)
    worksheet = writer.sheets[sheet_name]
    
    # Format numbers with commas and no decimal places
    number_format = '#,##0'
    
    # Get the starting row and column for the data
    start_row = 2  # Assuming header is in row 1
    
    # Format numeric columns
    for col_idx, col_name in enumerate(df.columns):
        if col_name in numeric_cols:
            col_letter = chr(65 + col_idx + start_col + 1)  # +1 because Excel columns start at A
            # Set column width to accommodate large numbers with commas
            worksheet.column_dimensions[col_letter].width = 20
            
            # Apply number format to all cells in the column
            for row in range(start_row, len(df) + start_row):
                cell = worksheet.cell(row=row, column=col_idx + start_col + 2)
                cell.number_format = number_format
                # Ensure the cell value is an integer
                if isinstance(cell.value, (int, float)):
                    cell.value = int(cell.value)

def main():
    db = DatabaseConnection()
    if db.connect():
        try:
            # Get 2025 data
            print("Fetching data for 2025...")
            cost_data = db.get_cost_data(year=2025)
            cash_data = db.get_cash_data(year=2025)
            
            # List of columns to analyze
            category_columns = [
                'EXPENDITURE_TYPE',
                'OWNER',
                'PROJECT_TYPE',
                'SECTOR',
                'AREAS'
            ]
            
            # Create Excel writer
            output_dir = '03.warehouse/11.cost.vs.cash'
            os.makedirs(output_dir, exist_ok=True)
            excel_path = os.path.join(output_dir, 'category_analysis_2025.xlsx')
            
            print(f"\nAnalyzing categories and exporting to {excel_path}")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # For each category column
                for col in category_columns:
                    print(f"\nAnalyzing by {col}...")
                    if col in cost_data.columns:
                        summary, detailed = analyze_by_category(cost_data, cash_data, col)
                        
                        # Export summary and detailed view side by side
                        export_to_excel(summary, col, writer, start_col=0)
                        export_to_excel(detailed, col, writer, start_col=len(summary.columns) + 2)
                        
                        print(f"Category: {col}")
                        print(f"Number of categories: {len(summary)}")
                        print(f"Total cost: {summary['Total_Cost'].sum():,.2f}")
                        print(f"Total cash: {summary['Total_Cash'].sum():,.2f}")
                    else:
                        print(f"Column {col} not found in cost data")
                
                # Create a sheet with unique values in each category
                print("\nCreating category values overview...")
                unique_values = pd.DataFrame()
                for col in category_columns:
                    if col in cost_data.columns:
                        unique_values[col] = pd.Series(cost_data[col].unique()).sort_values()
                
                export_to_excel(unique_values, 'Category_Values', writer)
            
            print("\nAnalysis complete!")
            
        finally:
            db.disconnect()

if __name__ == "__main__":
    main() 