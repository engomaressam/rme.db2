from db_connection import DatabaseConnection
import pandas as pd
import os

def get_top_line_descriptions(df, vendor_name, n=4):
    """Get the n most common line descriptions for a vendor"""
    if 'LINE_DESC' not in df.columns or df.empty:
        return [''] * n
    
    # Get line descriptions for this vendor
    vendor_lines = df[df['VENDOR_NAME'] == vendor_name]['LINE_DESC'].dropna()
    
    # Count unique descriptions and get top n
    desc_counts = vendor_lines.value_counts()
    top_descs = desc_counts.head(n)
    
    # Pad with empty strings if less than n descriptions
    result = list(top_descs.index)
    result.extend([''] * (n - len(result)))
    return result

def analyze_suppliers():
    db = DatabaseConnection()
    if db.connect():
        try:
            # Get 2025 data
            print("Fetching data for 2025...")
            cost_data = db.get_cost_data(year=2025)
            cash_data = db.get_cash_data(year=2025)
            
            # Calculate totals per vendor
            print("\nCalculating vendor totals...")
            cost_totals = cost_data.groupby('VENDOR_NAME')['AMOUNT'].sum()
            cash_totals = cash_data.groupby('Supplier Name')['AMOUNT'].sum()
            
            # Get unique vendors from both datasets
            all_vendors = pd.Index(set(cost_data['VENDOR_NAME'].unique()) | 
                                 set(cash_data['Supplier Name'].unique()))
            
            # Create results DataFrame
            results = pd.DataFrame(index=all_vendors)
            
            # Add line descriptions
            print("\nAnalyzing vendor line descriptions...")
            for i in range(1, 5):  # Get top 4 line descriptions
                results[f'Item_{i}'] = results.index.map(
                    lambda x: get_top_line_descriptions(cost_data, x, 4)[i-1]
                )
            
            # Add totals
            results['Total_Cost_2025'] = pd.to_numeric(cost_totals.reindex(all_vendors).fillna(0), errors='coerce')
            results['Total_Cash_2025'] = pd.to_numeric(cash_totals.reindex(all_vendors).fillna(0), errors='coerce')
            
            # Sort by total cost descending
            results = results.sort_values('Total_Cost_2025', ascending=False)
            
            # Export to Excel
            output_dir = '03.warehouse/11.cost.vs.cash'
            os.makedirs(output_dir, exist_ok=True)
            excel_path = os.path.join(output_dir, 'supplier_type_analysis_2025.xlsx')
            
            print(f"\nExporting results to {excel_path}")
            
            # Convert numeric columns to float and format them
            for col in ['Total_Cost_2025', 'Total_Cash_2025']:
                results[col] = pd.to_numeric(results[col], errors='coerce')
                # Format with commas and 2 decimal places
                results[col] = results[col].apply(lambda x: '{:,.2f}'.format(x) if pd.notnull(x) else 0)
                # Convert back to float
                results[col] = pd.to_numeric(results[col].str.replace(',', ''))
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                results.to_excel(writer, sheet_name='Supplier Analysis')
                
                # Get the workbook and worksheet
                workbook = writer.book
                worksheet = writer.sheets['Supplier Analysis']
                
                # Format numbers
                number_format = '#,##0.00'
                for row in range(2, len(results) + 2):
                    for col in ['F', 'G']:  # Total_Cost and Total_Cash columns
                        cell = worksheet[f'{col}{row}']
                        cell.number_format = number_format
                
                # Adjust column widths
                worksheet.column_dimensions['A'].width = 50  # Vendor name
                for col in ['B', 'C', 'D', 'E']:  # Line description columns
                    worksheet.column_dimensions[col].width = 40
                worksheet.column_dimensions['F'].width = 20  # Total cost
                worksheet.column_dimensions['G'].width = 20  # Total cash
            
            # Print some statistics
            print("\nAnalysis complete!")
            print(f"Total unique suppliers analyzed: {len(results)}")
            print(f"Total cost in 2025: {results['Total_Cost_2025'].sum():,.2f}")
            print(f"Total cash in 2025: {results['Total_Cash_2025'].sum():,.2f}")
            
        finally:
            db.disconnect()

if __name__ == "__main__":
    analyze_suppliers() 