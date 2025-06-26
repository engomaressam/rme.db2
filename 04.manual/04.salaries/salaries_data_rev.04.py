
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import weasyprint
from weasyprint import CSS


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


# Initialize the Dash app
app = dash.Dash(__name__)

# Define a function to calculate the total salary amount for a selected project
def calculate_total_salary(selected_project):
    filtered_df = df[df['project'] == selected_project]
    total_salary = filtered_df['amount'].sum()
    return total_salary


# Define the layout of the web app
app.layout = html.Div([
    dcc.Dropdown(
        id='project-dropdown',
        options=[{'label': project, 'value': project} for project in df['project'].unique()],
        value=df['project'].unique()[0]
    ),
    html.Div([
        html.Div(id='total-salary-text', style={'fontSize': 24, 'fontWeight': 'bold', 'textAlign': 'center'}),
    ]),
    dcc.Graph(id='salary-line-plot')
])


# Define the callback to update the plot and total salary text based on the selected project
@app.callback(
    [Output('salary-line-plot', 'figure'),
     Output('total-salary-text', 'children')],
    Input('project-dropdown', 'value')
)
def update_graph(selected_project):
    filtered_df = df[df['project'] == selected_project]
    total_salary = calculate_total_salary(selected_project)
    fig = px.line(filtered_df, x='month', y='amount', title=f'Salaries for {selected_project}')
    total_salary_text = f"Total Salary for {selected_project}: {total_salary:,.2f} EGP"
    return fig, total_salary_text

if __name__ == '__main__':
    # Read the Excel file into a Pandas DataFrame
    df = pd.read_excel("Salaries RME (Nov-15 To Sep-23).xlsx")

    # Run the Dash app locally
    app.run_server(debug=True)

    # Save each plot to a PDF file
    for project in df['project'].unique():
        filtered_df = df[df['project'] == project]
        fig = px.line(filtered_df, x='month', y='amount', title=f'Salaries for {project}')
        fig.write_image(f"{project}_plot.png")

    # Combine all plots into a single PDF
    pdf_pages = []
    for project in df['project'].unique():
        pdf_pages.append(weasyprint.Image(f"{project}_plot.png"))

    weasyprint.pdf_pages_to_pdf_file(pdf_pages, "plots.pdf", stylesheets=[CSS(string="body { text-align: center; }")])


import weasyprint
from PIL import Image
from PyPDF2 import PdfMerger

# Create a list to store PDF pages
pdf_pages = []

# Iterate through each project and create a page for each
for project in df['project'].unique():
    image_path = f"{project}_plot.png"

    try:
        # Open the image file
        im = Image.open(image_path)
        im = im.rotate(90, expand=True)
        im.save(image_path)

        # Create an HTML page with image and additional information
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        text-align: center;
                    }}
                </style>
            </head>
            <body>
                <h1>Project: {project}</h1>
                <img src="{image_path}" alt="Plot Image" width="400" height="300">
                <p>Total Salary: {calculate_total_salary(project):,.2f} EGP</p>
        """

        # Check if there is data available for the latest month
        latest_month_amount = get_latest_month_amount(project)
        if latest_month_amount is not None:
            html_content += f"<p>Last Month's Salary: {latest_month_amount:,.2f} EGP</p>"

        html_content += """
            </body>
        </html>
        """

        # Convert HTML to PDF using weasyprint
        pdf_data = weasyprint.HTML(string=html_content).write_pdf()

        # Append the PDF page to the list
        pdf_pages.append(pdf_data)

    except Exception as e:
        print(f"Error adding page for {project}: {e}")

# Merge all PDF pages into a single PDF file using PdfMerger
pdf_merger = PdfMerger()
for pdf_data in pdf_pages:
    pdf_merger.append(fileobj=pdf_data)

# Save the merged PDF to a file
with open("plots.pdf", "wb") as pdf_file:
    pdf_merger.write(pdf_file)



