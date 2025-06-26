# %%
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State  # Import dcc and html from dash
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

# %%
# Read the Excel file
df = pd.read_excel("staff.xlsx")

# Read the project types
df_types = pd.read_excel("types.xlsx")

# Read the Departments Data
df_departments = pd.read_excel("departments.xlsx")

#Combine df (containing Job, Project, and Month) with df_departments (containing Job and Department) to create a single DataFrame with all necessary information:
df = df.merge(df_departments, on='Job', how='left')

# Create dropdown menu options from unique project names
project_options = [{'label': project, 'value': project} for project in df['Project'].unique()]

# Create dropdown menu options for project types
type_options = [{'label': project_type, 'value': project_type} for project_type in df_types['Type'].unique()]

# Create dropdown menu options for departments
department_options = [{'label': 'Select All', 'value': 'all'}] + [
    {'label': department, 'value': department} for department in df_departments['Department'].unique()
]


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# %%
# Define the layout of the app
app.layout = html.Div([
    html.H1("Deployment Plan"),
    dbc.Button("New Project", id="open", color="primary", className="mr-1"),
    dbc.Modal(
        [
            dbc.ModalHeader("New Project Details"),
            dbc.ModalBody(
                [
                    dcc.Input(id='type-input', type='text', placeholder='Type of Project'),
                    dcc.Input(id='duration-input', type='text', placeholder='Duration of Project')
                ]
            ),
            dbc.ModalFooter(
                [
                    dbc.Button("Close", id="close", className="ml-auto"),
                    dbc.Button("Save Changes", id="save-changes", className="ml-1")
                ]
            ),
        ],
        id="modal",
        size='lg',
        centered=True,
        backdrop='static'
    ),
    dcc.Dropdown(
        id='type-dropdown',
        options=type_options,
        value=type_options[0]['value'] if type_options else None  # Default value
    ),
    dcc.Dropdown(
        id='project-dropdown',
        options=project_options,
        value=project_options[0]['value']  # Default value
    ),
    dcc.Dropdown(
        id='department-dropdown',
        options=department_options,
        value='all'  # Default to 'Select All'
    ),
    dcc.Html(id='deployment-plan'),  # Use dcc.Html for the table
    html.Div(id='selected-project-type')  # Show the selected project type
])


# %%
# Define callback to update the deployment plan based on selected project and department
@app.callback(
    Output('deployment-plan', 'children'),
    [Input('project-dropdown', 'value'),
     Input('department-dropdown', 'value')]
)
def update_plan(selected_project, selected_department):
    if selected_project is None:
        return []  # Handle initial case

    # Filter data based on project and department
    if selected_department == 'all':
        filtered_data = df[df['Project'] == selected_project]  # Filter only by project
    else:
        filtered_data = df[(df['Project'] == selected_project) & (df['Department'] == selected_department)]

    if filtered_data.empty:
        return []  # Handle empty data

    # Get unique job titles and months
    unique_jobs = filtered_data['Job'].unique()
    months = sorted(filtered_data['Month'].unique())

    # Create a pivot table to summarize staff count by job and month
    staff_table = filtered_data.pivot_table(
        index='Job',
        columns='Month',
        values='Job',
        aggfunc='count'
    )

    # Fill missing values with 0 (optional)
    staff_table.fillna(0, inplace=True)

    # Convert the pivot table to HTML table using `to_html` method
    html_table = staff_table.to_html(index=True, justify='center')

    return html_table

# %%
# New callback for the modal popup
@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks"), Input("save-changes", "n_clicks")],
    [dash.dependencies.State("modal", "is_open")],
)
def toggle_modal(n1, n2, n3, is_open):
    if n1 or n2 or n3:
        return not is_open
    return is_open


# %%
# Callback to show the project type for the selected project
@app.callback(
    Output('selected-project-type', 'children'),
    [Input('project-dropdown', 'value')]
)
def display_project_type(selected_project):
    # Get the type of the selected project
    project_type = df_types[df_types['Project'] == selected_project]['Type'].values[0]
    return f'Type of Project: {project_type}'


# %%
if __name__ == '__main__':
    app.run_server(debug=True)


