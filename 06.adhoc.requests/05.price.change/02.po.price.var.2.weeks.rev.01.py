# %%
import pandas as pd
import glob

# %%
# Step 1: Find and read the only xlsx file in the directory
file_list = glob.glob('*.xlsx')
if len(file_list) != 1:
    raise ValueError("There should be exactly one .xlsx file in the directory.")
file_path = file_list[0]

df = pd.read_excel(file_path)

# %%
# Step 2: Extract relevant columns including the additional ones
df = df[['description', 'unit', 'unit_price', 'approved_date', 'project_name', 'vendor', 'qty', 'amount_egp', 'project_no', 'organization_code', 'buyer_dept', 'buyer', 'qty_received']]

# %%
# Step 3: Convert 'approved_date' to datetime and determine the week number
df['approved_date'] = pd.to_datetime(df['approved_date'])
df['week'] = df['approved_date'].dt.isocalendar().week
df['year'] = df['approved_date'].dt.isocalendar().year  # Add year to handle year-end week changes


# %%
# Step 4: Group by 'description', 'unit', 'year', and 'week'
grouped = df.groupby(['description', 'unit', 'year', 'week'])

# %%
# Step 5 & 6: Check for variations in 'unit_price' between consecutive weeks
results = []

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

# %%
# Combine all results into a single DataFrame
if results:
    result_df = pd.concat(results).drop_duplicates()
else:
    result_df = pd.DataFrame()

# %%
# Output the result
result_df.to_excel('output.xlsx', index=False)

print("Analysis complete. Results are saved in 'output.xlsx'.")


