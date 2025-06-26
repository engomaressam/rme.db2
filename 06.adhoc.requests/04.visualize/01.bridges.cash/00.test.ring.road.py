# Import necessary libraries
import pandas as pd
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource, Select, HoverTool, PrintfTickFormatter
from bokeh.layouts import column
from bokeh.palettes import Category20
from bokeh.transform import factor_cmap
from bokeh.models.tickers import FixedTicker
from bokeh.io import curdoc
from bokeh.events import ButtonClick

# Read the Excel file with two sheets "In Values" and "Out Values"
# Update the file path and sheet names as needed
excel_file = "D:/OneDrive/OneDrive - Rowad Modern Engineering/x004 Data Science/03.rme.db/00.repo/rme.db/10.visualize/01.bridges.cash/00.main.file/Bridges Sector Cash In And Out Data.xlsx"
in_values_df = pd.read_excel(excel_file, sheet_name="In Values")
out_values_df = pd.read_excel(excel_file, sheet_name="Out Values")

# Filter the data for the project "0122 Ring Road"
selected_project = "0122 Ring Road"
filtered_data = in_values_df[in_values_df['Project Name'] == selected_project]

# Create a list of unique projects for the dropdown menu
projects = in_values_df['Project Name'].unique().tolist()

# Create a ColumnDataSource for the filtered data
source = ColumnDataSource(data=filtered_data)

# Convert the 'Date' column to the 'Mon-YY' format
filtered_data['Date'] = filtered_data['Date'].dt.strftime('%b-%y').astype(str)

print(filtered_data['Date'].unique())

# Assuming 'filtered_data' has the 'Date' column in 'Mon-YY' format
x_range = filtered_data['Date'].tolist()

# Create a Bokeh figure using the x_range
p = figure(x_range=x_range, height=350, title="Cash In per Month")

# Create a custom palette for different months
color_mapper = factor_cmap(field_name='Date', palette=Category20[len(filtered_data['Date'].unique())], factors=filtered_data['Date'].unique())

# Create a bar chart for cash in with the custom palette
p.vbar(x='Date', top='Amount', source=source, width=0.8, color=color_mapper)

# Configure the plot
p.xaxis.major_label_orientation = "vertical"
p.xaxis.axis_label = "Date"
p.yaxis.axis_label = "Amount"
p.yaxis[0].formatter = PrintfTickFormatter(format="$%.0f")

# Create a hover tool to display values on hover
hover = HoverTool()
hover.tooltips = [("Date", "@Date"), ("Amount", "@Amount{$0}")]
p.add_tools(hover)

# Create a dropdown menu to select the project
project_select = Select(title="Select Project:", options=projects, value=selected_project)

# Create a callback function to update the data source based on the selected project
def update_project(attrname, old_project, new_project):
    selected_project = project_select.value
    filtered_data = in_values_df[in_values_df['Project Name'] == selected_project]
    source.data = ColumnDataSource.from_df(filtered_data)

# Attach the callback function to the project dropdown menu
project_select.on_change('value', update_project)

# Create a layout for the plot and dropdown menu
layout = column(project_select, p)

# Add the layout to the current document
curdoc().add_root(layout)