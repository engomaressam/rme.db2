
import pandas as pd

# Specify the file path
file_path = "Salaries RME (Nov-15 To Sep-23).xlsx"

# Read the Excel file into a Pandas DataFrame
df = pd.read_excel(file_path)

# Group the data by the 'project' column and sum the 'amount' for each project
project_totals = df.groupby('project')['amount'].sum().reset_index()

# Specify the output Excel file path
output_file = "Project_Salary_Totals.xlsx"

# Save the project totals to a new Excel file
project_totals.to_excel(output_file, index=False)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Define a function to calculate the total salary amount for a selected project
def calculate_total_salary(selected_project):
    filtered_df = df[df['project'] == selected_project]
    total_salary = filtered_df['amount'].sum()
    return total_salary

# Define a function to get the amount of the latest month for a selected project
def get_latest_month_amount(selected_project):
    filtered_df = df[df['project'] == selected_project]
    latest_month = filtered_df['month'].max()
    latest_amount = filtered_df[filtered_df['month'] == latest_month]['amount'].values[0]
    return latest_amount

# Define the layout of the web app
app.layout = html.Div([
    dcc.Dropdown(
        id='project-dropdown',
        options=[{'label': project, 'value': project} for project in df['project'].unique()],
        value=df['project'].unique()[0]
    ),
    html.Div([
        html.Div(id='total-salary-text', style={'fontSize': 24, 'fontWeight': 'bold', 'textAlign': 'center'}),
        html.Div(id='latest-month-amount-text', style={'fontSize': 18, 'fontWeight': 'bold', 'textAlign': 'center'}),
    ]),
    dcc.Graph(id='salary-line-plot')
])

# Define the callback to update the plot and total salary text based on the selected project
@app.callback(
    [Output('salary-line-plot', 'figure'),
     Output('total-salary-text', 'children'),
     Output('latest-month-amount-text', 'children')],
    Input('project-dropdown', 'value')
)
def update_graph(selected_project):
    filtered_df = df[df['project'] == selected_project]
    total_salary = calculate_total_salary(selected_project)
    latest_month_amount = get_latest_month_amount(selected_project)
    fig = px.line(filtered_df, x='month', y='amount', title=f'Salaries for {selected_project}')
    total_salary_text = f"Total Salary for {selected_project}: {total_salary:,.2f} EGP"
    latest_month_amount_text = f"Amount for Latest Month: {latest_month_amount:,.2f} EGP"
    return fig, total_salary_text, latest_month_amount_text

if __name__ == '__main__':
    app.run_server(debug=True)


