import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Select
from bokeh.io import curdoc, output_notebook, output_file, push_notebook
from bokeh.transform import factor_cmap
from bokeh.layouts import layout
from bokeh.palettes import Viridis256  # Import Viridis256 for a large number of colors
from bokeh.models import FactorRange  # Import FactorRange for categorical x-range

# Read the Excel file with two sheets "In Values" and "Out Values"
# Update the file path and sheet names as needed
excel_file = "D:/OneDrive/OneDrive - Rowad Modern Engineering/x004 Data Science/03.rme.db/00.repo/rme.db/10.visualize/01.bridges.cash/00.main.file/Bridges Sector Cash In And Out Data.xlsx"
in_values_df = pd.read_excel(excel_file, sheet_name="In Values")
out_values_df = pd.read_excel(excel_file, sheet_name="Out Values")

# Convert the 'Date' column to datetime type if it's not already
in_values_df['Date'] = pd.to_datetime(in_values_df['Date'])

# Extract year and month from the 'Date' column
in_values_df['Year'] = in_values_df['Date'].dt.year
in_values_df['Month'] = in_values_df['Date'].dt.month_name()

# Remove rows with NaN values in the 'Month' column
in_values_df.dropna(subset=['Month'], inplace=True)

# Create a Bokeh data source from the aggregated DataFrame
source = ColumnDataSource(data=in_values_df)

# Define the Bokeh figure with a FactorRange for the x-range
p = figure(x_range=FactorRange(*in_values_df['Month'].unique()), height=350, title="Cash In per Month")

# Create a bar chart for cash in with dynamically generated colors
color_mapper = factor_cmap(field_name='Month', palette=Viridis256, factors=in_values_df['Month'].unique())
p.vbar(x='Month', top='Amount', source=source, width=0.8, color=color_mapper)

# Configure the plot
p.xaxis.major_label_orientation = 45  # Rotate x-axis labels for readability

# Create a Select widget to choose a project
project_select = Select(title="Select Project:", options=in_values_df['Project Name'].unique().tolist())

# Define a callback function to update the plot based on the selected project
def update_plot(attrname, old_value, new_value):
    selected_project = project_select.value
    new_data = in_values_df[in_values_df['Project Name'] == selected_project]
    
    # Update the data source for the plot
    source.data = {
        'Month': new_data['Month'],
        'Amount': new_data['Amount']
    }
    push_notebook()

project_select.on_change('value', update_plot)

# Create a layout for the plot and Select widget
layout = layout([[project_select], [p]])

# Save the plot as an HTML file (optional)
output_file("cash_flow_plot.html")

# Show the plot in the notebook
output_notebook()
show(layout, notebook_handle=True)
