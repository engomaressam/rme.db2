import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Read the average job counts from the Excel file
average_job_counts_by_type = pd.read_excel("average_job_counts_by_type.xlsx")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Take input from the user for the type of project and its duration
        project_type = request.form['project_type']
        duration = int(request.form['duration'])

        # Determine the start month (Mar-24) and end month based on the duration
        start_month = pd.to_datetime('2024-03-01')
        end_month = start_month + pd.DateOffset(months=duration - 1)

        # Create a list of months from start to end
        months = pd.date_range(start=start_month, end=end_month, freq='MS')

        # Initialize the deployment plan DataFrame
        deployment_plan = pd.DataFrame({'Month': months})

        # Calculate the middle month index (for the start of the average period)
        middle_month_index = len(deployment_plan) // 2

        # Define the pattern for increasing and decreasing counts
        increasing_pattern = list(range(middle_month_index))
        decreasing_pattern = list(range(middle_month_index, 0, -1))

        # Calculate the number of people for each job in each month
        for job in sorted(average_job_counts_by_type['Job']):
            # Handle the special case for "Project Manager"
            if job == "Project Manager":
                counts = [1] * len(deployment_plan)
            else:
                # Calculate the average count for the job
                avg_count = average_job_counts_by_type.loc[average_job_counts_by_type['Job'] == job, 'Average Count'].iloc[0]

                # Initialize counts list for the job
                counts = []

                # Generate counts based on the pattern
                for i in range(len(deployment_plan)):
                    if i < middle_month_index:
                        index = i % len(increasing_pattern)  # Ensure index is within the range of the pattern list
                        counts.append(int(round((increasing_pattern[index] / middle_month_index) * avg_count)))
                    elif i == middle_month_index:
                        counts.append(int(avg_count))
                    else:
                        index = (i - middle_month_index) % len(decreasing_pattern)  # Ensure index is within the range of the pattern list
                        counts.append(int(round((decreasing_pattern[index] / middle_month_index) * avg_count)))

            # Add counts to the deployment plan DataFrame
            deployment_plan[job] = counts

        # Transpose the DataFrame to swap rows and columns
        deployment_plan_transposed = deployment_plan.set_index('Month').T

        # Pass the transposed DataFrame to the HTML template
        return render_template('index.html', deployment_plan_html=deployment_plan_transposed.to_html())
    else:
        # Render the form template if no data has been submitted
        return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
