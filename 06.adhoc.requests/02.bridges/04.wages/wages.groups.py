import pandas as pd

# Specify the file path to your Excel file
file_path = "Wages 2023 Till Sep.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Extract the required columns: 'occupation', 'project', and 'amount'
df = df[['occupation', 'project', 'amount']]

# Group by 'project' and calculate the sum of 'amount' for each project
project_wages = df.groupby('project')['amount'].sum().reset_index()

# Specify the output Excel file for the summed data
output_file = "Summed_Wages_By_Project.xlsx"

# Save the summed data to a new Excel file
project_wages.to_excel(output_file, index=False)

# Group by both 'project' and 'occupation' and calculate the sum of 'amount' for each combination
occupation_project_wages = df.groupby(['project', 'occupation'])['amount'].sum().reset_index()

# Specify the output Excel file for the summed data
output_file = "Summed_Wages_By_Occupation_By_Project.xlsx"

# Save the summed data to a new Excel file
occupation_project_wages.to_excel(output_file, index=False)

# Define the list of keywords to filter for
direct_keywords = ["نجار", "شده", "خرسانة", "فورمجى", "حداد"]

# Create a mask to filter direct costs
direct_cost_mask = df['occupation'].str.contains('|'.join(direct_keywords), case=False, na=False)

# Calculate direct costs by summing the amounts for direct cost rows
direct_cost_df = df[direct_cost_mask].groupby('project')['amount'].sum().reset_index()
direct_cost_df.rename(columns={'amount': 'direct_cost'}, inplace=True)

# Calculate indirect costs by subtracting direct costs from the total amounts
indirect_cost_df = df.groupby('project')['amount'].sum().reset_index()
indirect_cost_df['indirect_cost'] = indirect_cost_df['amount'] - indirect_cost_df['project'].map(direct_cost_df.set_index('project')['direct_cost'])

# Specify the output Excel file for indirect cost data
output_file = "Indirect_Cost_Wages_By_Project.xlsx"

# Save the indirect cost data to a new Excel file
indirect_cost_df.to_excel(output_file, index=False)

# Specify the output Excel file for direct cost data
output_file = "Direct_Cost_Wages_By_Project.xlsx"

# Save the indirect cost data to a new Excel file
direct_cost_df.to_excel(output_file, index=False)


