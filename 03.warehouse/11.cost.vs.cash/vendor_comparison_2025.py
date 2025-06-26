import mysql.connector as mysql
import pandas as pd
import os

def fetch_data(cursor, query, columns):
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=columns)
    df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
    return df

def main():
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

        # Fetch ALL cost data for 2025 (no transaction type filter)
        print("Fetching cost data...")
        cost_query = """
        SELECT 
            VENDOR_NAME,
            AMOUNT,
            GL_DATE,
            TRANSACTION_SOURCE
        FROM RME_Projects_Cost_Dist_Line_Report
        WHERE GL_DATE LIKE '2025%'
        """
        cost_df = fetch_data(cursor, cost_query, ['VENDOR_NAME', 'AMOUNT', 'GL_DATE', 'TRANSACTION_SOURCE'])
        print(f"Fetched {len(cost_df)} cost records")

        # Fetch ALL cash data for 2025
        print("Fetching cash data...")
        cash_query = """
        SELECT 
            `Supplier Name` as VENDOR_NAME,
            AMOUNT,
            CHECK_DATE
        FROM RME_ap_check_payments_Report
        WHERE YEAR(CHECK_DATE) = 2025
        """
        cash_df = fetch_data(cursor, cash_query, ['VENDOR_NAME', 'AMOUNT', 'CHECK_DATE'])
        print(f"Fetched {len(cash_df)} cash records")

        # Calculate totals by vendor
        cost_vendor_totals = cost_df.groupby('VENDOR_NAME')['AMOUNT'].sum()
        cash_vendor_totals = cash_df.groupby('VENDOR_NAME')['AMOUNT'].sum()

        # Get all unique vendors
        all_vendors = sorted(set(cost_vendor_totals.index) | set(cash_vendor_totals.index))

        # Create comprehensive comparison DataFrame
        comparison_df = pd.DataFrame(index=all_vendors)
        comparison_df['Cost Amount'] = cost_vendor_totals
        comparison_df['Cash Amount'] = cash_vendor_totals
        comparison_df = comparison_df.fillna(0)
        comparison_df['Difference (Cost - Cash)'] = comparison_df['Cost Amount'] - comparison_df['Cash Amount']
        
        # Identify vendors with only cost or only cash
        cost_only_vendors = comparison_df[
            (comparison_df['Cost Amount'] != 0) & 
            (comparison_df['Cash Amount'] == 0)
        ].index.tolist()

        cash_only_vendors = comparison_df[
            (comparison_df['Cost Amount'] == 0) & 
            (comparison_df['Cash Amount'] != 0)
        ].index.tolist()

        # Create DataFrames for vendors with only cost or only cash
        cost_only_df = comparison_df.loc[cost_only_vendors, ['Cost Amount']].sort_values('Cost Amount', ascending=False)
        cash_only_df = comparison_df.loc[cash_only_vendors, ['Cash Amount']].sort_values('Cash Amount', ascending=False)

        # Sort the main comparison by absolute difference
        comparison_df = comparison_df.sort_values('Difference (Cost - Cash)', key=abs, ascending=False)

        # Save to Excel with multiple sheets
        output_path = '03.warehouse/11.cost.vs.cash/vendor_comparison_2025.xlsx'
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            comparison_df.to_excel(writer, sheet_name='All Vendors Comparison')
            cost_only_df.to_excel(writer, sheet_name='Cost Only Vendors')
            cash_only_df.to_excel(writer, sheet_name='Cash Only Vendors')

            # Add summary statistics
            summary_data = {
                'Metric': [
                    'Total Number of Vendors',
                    'Vendors with Both Cost and Cash',
                    'Vendors with Only Cost',
                    'Vendors with Only Cash',
                    'Total Cost Amount',
                    'Total Cash Amount',
                    'Net Difference (Cost - Cash)'
                ],
                'Value': [
                    len(all_vendors),
                    len(all_vendors) - len(cost_only_vendors) - len(cash_only_vendors),
                    len(cost_only_vendors),
                    len(cash_only_vendors),
                    comparison_df['Cost Amount'].sum(),
                    comparison_df['Cash Amount'].sum(),
                    comparison_df['Difference (Cost - Cash)'].sum()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"\nAnalysis saved to: {output_path}")
        print("\nSummary:")
        print(f"Total vendors: {len(all_vendors)}")
        print(f"Vendors with only cost: {len(cost_only_vendors)}")
        print(f"Vendors with only cash: {len(cash_only_vendors)}")

    except mysql.Error as error:
        print(f"Error connecting to database: {error}")

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals():
            connection.close()
            print("\nMySQL connection is closed")

if __name__ == "__main__":
    main() 