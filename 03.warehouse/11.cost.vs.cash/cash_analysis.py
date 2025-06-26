import mysql.connector as mysql
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Create output directory if it doesn't exist
    output_dir = '03.warehouse/11.cost.vs.cash'
    os.makedirs(output_dir, exist_ok=True)

    # Connection details
    host = "10.10.11.242"
    user = "omar2"
    password = "Omar_54321"
    database = "RME_TEST"

    try:
        print("Connecting to database...")
        connection = mysql.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        if connection.is_connected():
            print("Connected to MySQL database")

        cursor = connection.cursor()

        print("Executing query...")
        query = """
        SELECT 
            PROJECT_NAME,
            `Supplier Name` as VENDOR_NAME,
            AMOUNT,
            CHECK_DATE
        FROM RME_ap_check_payments_Report
        WHERE YEAR(CHECK_DATE) = 2025
        """
        
        cursor.execute(query)
        print("Fetching data...")
        data = cursor.fetchall()
        print(f"Fetched {len(data)} rows")
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['PROJECT_NAME', 'VENDOR_NAME', 'AMOUNT', 'CHECK_DATE'])
        print("\nDataFrame shape:", df.shape)
        
        # Convert AMOUNT to numeric
        df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
        
        # Group by Project and calculate total amount
        project_totals = df.groupby('PROJECT_NAME')['AMOUNT'].sum().sort_values(ascending=True)
        top_10_projects = project_totals.tail(10)
        
        # Group by Vendor and calculate total amount
        vendor_totals = df.groupby('VENDOR_NAME')['AMOUNT'].sum().sort_values(ascending=True)
        top_10_vendors = vendor_totals.tail(10)
        
        print("\nCreating visualizations...")
        
        # Projects plot
        plt.figure(figsize=(15, 8))
        plt.barh(range(len(top_10_projects)), top_10_projects.values/1_000_000)
        plt.title('Top 10 Projects by Cash Payments (2025)')
        plt.xlabel('Amount (Millions EGP)')
        plt.ylabel('Project Name')
        plt.yticks(range(len(top_10_projects)), top_10_projects.index)
        
        # Add value labels
        for i, v in enumerate(top_10_projects.values):
            plt.text(v/1_000_000, i, f'{v/1_000_000:,.1f}M', va='center')
        
        # Save projects plot
        projects_plot_path = os.path.join(output_dir, 'top_projects_cash_2025.png')
        plt.savefig(projects_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Projects plot saved to: {projects_plot_path}")
        
        # Vendors plot
        plt.figure(figsize=(15, 8))
        plt.barh(range(len(top_10_vendors)), top_10_vendors.values/1_000_000)
        plt.title('Top 10 Vendors by Cash Payments (2025)')
        plt.xlabel('Amount (Millions EGP)')
        plt.ylabel('Vendor Name')
        plt.yticks(range(len(top_10_vendors)), top_10_vendors.index)
        
        # Add value labels
        for i, v in enumerate(top_10_vendors.values):
            plt.text(v/1_000_000, i, f'{v/1_000_000:,.1f}M', va='center')
        
        # Save vendors plot
        vendors_plot_path = os.path.join(output_dir, 'top_vendors_cash_2025.png')
        plt.savefig(vendors_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Vendors plot saved to: {vendors_plot_path}")
        
        # Print summaries
        print("\nTop 10 Projects by Cash Payments:")
        print(top_10_projects.to_frame().to_string())
        
        print("\nTop 10 Vendors by Cash Payments:")
        print(top_10_vendors.to_frame().to_string())
        
        # Save to Excel
        excel_path = os.path.join(output_dir, 'cash_analysis_2025.xlsx')
        with pd.ExcelWriter(excel_path) as writer:
            top_10_projects.to_frame().to_excel(writer, sheet_name='Top Projects')
            top_10_vendors.to_frame().to_excel(writer, sheet_name='Top Vendors')
        print(f"\nDetailed data saved to: {excel_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("\nMySQL connection is closed")

if __name__ == "__main__":
    main() 