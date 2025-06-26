import pandas as pd
from flask import Flask, render_template, request
import os
import webbrowser
from threading import Timer

app = Flask(__name__)

# Read the average job counts from the Excel file
average_job_counts_by_type = pd.read_excel("average_job_counts_by_type.xlsx")

def generate_deployment_plan(project_type, duration):
    # Filter the average job counts by the selected project type
    average_job_counts_filtered = average_job_counts_by_type[average_job_counts_by_type['Type'] == project_type]

    # Extract the 'Job' column
    job_column = average_job_counts_filtered['Job'] if 'Job' in average_job_counts_filtered else None

    # Determine the start month (Mar-24) and end month based on the duration
    start_month = pd.to_datetime('2024-03-01')
    end_month = start_month + pd.DateOffset(months=duration - 1)

    # Create a list of months from start to end
    months = pd.date_range(start=start_month, end=end_month, freq='MS')

    # Initialize the list to hold row dictionaries
    rows = []

    # Calculate the middle month index (for the start of the average period)
    middle_month_index = len(months) // 2

    # Define the pattern for increasing and decreasing counts
    increasing_pattern = list(range(middle_month_index))
    decreasing_pattern = list(range(middle_month_index, -1, -1))  # Modified to include 0

    # Calculate the number of people for each job in each month
    for month in months:
        month_counts = {'Month': month}
        for job in sorted(job_column):
            # Handle the special case for "Project Manager"
            if job == "Project Manager":
                month_counts[job] = 1
            elif job == "Surveying":
                # Calculate the peak count for Surveying
                peak_count = max([round(average_job_counts_filtered.loc[average_job_counts_filtered['Job'] == job, 'Average Count'].iloc[0] * (i / middle_month_index)) for i in range(1, middle_month_index + 1)])
                # Calculate counts for Surveying
                if month < months[middle_month_index]:
                    month_counts[job] = int(1 + (peak_count - 1) * months.get_loc(month) / middle_month_index)  # Increase gradually to peak
                elif month == months[middle_month_index]:
                    month_counts[job] = peak_count  # Maintain peak in the middle
                else:
                    month_counts[job] = int(peak_count - (peak_count - 1) * (months.get_loc(month) - middle_month_index) / middle_month_index)  # Decrease gradually to 1
            else:
                # Calculate the average count for the job
                avg_count = average_job_counts_filtered.loc[average_job_counts_filtered['Job'] == job, 'Average Count'].iloc[0]

                # Generate counts based on the pattern
                if months.get_loc(month) < middle_month_index:
                    index = months.get_loc(month) % len(increasing_pattern)  # Ensure index is within the range of the pattern list
                    month_counts[job] = int(round((increasing_pattern[index] / middle_month_index) * avg_count))
                elif months.get_loc(month) == middle_month_index:
                    month_counts[job] = int(avg_count)
                else:
                    index = (months.get_loc(month) - middle_month_index) % len(decreasing_pattern)  # Ensure index is within the range of the pattern list
                    month_counts[job] = int(round((decreasing_pattern[index] / middle_month_index) * avg_count))
        rows.append(month_counts)

    # Convert the list of row dictionaries to a DataFrame
    deployment_plan = pd.DataFrame(rows)

    # Convert the 'Month' column to the desired format
    deployment_plan['Month'] = months.strftime('%b-%y')

    # Set 'Month' column as index
    deployment_plan.set_index('Month', inplace=True)

    # Drop the 'Job' column if it exists
    if 'Job' in deployment_plan.columns:
        deployment_plan.drop(columns=['Job'], inplace=True)

    return deployment_plan


@app.route('/', methods=['GET', 'POST'])
def index():
    bridge_length = None  # Initialize bridge_length variable

    if request.method == 'POST':
        # Take input from the user for the type of project, its duration, and bridge length if applicable
        project_type = request.form['project_type']
        duration = int(request.form['duration'])

        # Generate the deployment plan
        deployment_plan = generate_deployment_plan(project_type, duration)

        # Transpose the DataFrame
        transposed_plan = deployment_plan.T

        # Define the desired order of job titles
        desired_order = ['Project Manager', 'Construction Manager', 'Site Engineer', 'Supervisor', 'Surveying', 'Technical Office']

        # Extract job titles that exist in the desired order
        existing_titles = [title for title in desired_order if title in transposed_plan.index]

        # Extract remaining job titles that are not in the desired order
        remaining_titles = [title for title in transposed_plan.index if title not in desired_order]

        # Concatenate the job titles in the desired order followed by the remaining titles
        ordered_index = existing_titles + sorted(remaining_titles)

        # Reindex the transposed DataFrame to reflect the desired row order
        transposed_plan = transposed_plan.reindex(ordered_index)
        
        # Pass the transposed DataFrame and bridge length to the HTML template
        return render_template('index.html', deployment_plan_html=transposed_plan.to_html())
    else:
        # Render the form template if no data has been submitted
        return render_template('form.html')

def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True)
