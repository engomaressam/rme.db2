# %%
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
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
server = app.server

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
    html.Div([
        html.Label('Type of Project:'),
        dcc.Dropdown(
            id='type-dropdown',
            options=type_options,
            value=type_options[0]['value'] if type_options else None  # Default value
        )
    ]),
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
    dcc.Graph(id='deployment-plan'),
    html.Div(id='selected-project-type')  # Show the selected project type
])


# %%
# Define callback to update the deployment plan based on selected project and department
@app.callback(
    Output('deployment-plan', 'figure'),
    [Input('project-dropdown', 'value'),
     Input('department-dropdown', 'value')]
)
def update_plan(selected_project, selected_department):
    if selected_project is None:
        # Handle the initial case where no project is selected
        return {'data': [], 'layout': {}}

    # Filter data based on project and department
    if selected_department == 'all':
        filtered_data = df[df['Project'] == selected_project]  # Filter only by project
    else:
        filtered_data = df[(df['Project'] == selected_project) & (df['Department'] == selected_department)]

    if filtered_data.empty:
        # Return an empty figure if the filtered data is empty
        return {'data': [], 'layout': {}}

    # Get unique job titles for the selected project
    unique_jobs = filtered_data['Job'].unique()

    # Get unique months for the selected project and sort them
    months = sorted(filtered_data['Month'].unique())

    # Create a matrix to store staff count for each job title in each month
    staff_matrix = []

    # Iterate through each job title and count staff for each month
    for job in unique_jobs:
        job_data = filtered_data[filtered_data['Job'] == job]
        staff_count = [
            job_data[job_data['Month'] == month].shape[0] if not job_data[job_data['Month'] == month].empty else None
            for month in months
        ]
        staff_matrix.append(staff_count)

    # Get the maximum staff count for the selected project
    max_value = max([item for sublist in staff_matrix for item in sublist if item is not None])

    # Create the heatmap figure
    fig = go.Figure(data=[
        go.Heatmap(
            z=staff_matrix,
            x=months,
            y=unique_jobs,
            colorscale='Viridis',  # You can choose a different colorscale if desired
            showscale=True,
            zmin=1,
            zmax=max_value,
            zmid=0,
            colorbar=dict(tickmode='array', tickvals=list(range(1, max_value + 1)), ticktext=list(range(1, max_value + 1))),
            reversescale=True  # Add this line to reverse the color scale
        )
    ])


    layout = go.Layout(
        title=f"Deployment Plan for {selected_project} ({selected_department})",
        xaxis=dict(title='Months'),
        yaxis=dict(title='Job Titles'),
    )

    return fig

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
import os
from threading import Timer
import webbrowser

def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:1222/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=1222)


