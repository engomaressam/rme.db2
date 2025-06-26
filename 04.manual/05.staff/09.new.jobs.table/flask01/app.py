from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Configure route for project selection form
@app.route("/")
def select_project():
    return render_template("forms.html")

# Route to return project data in JSON format
@app.route("/data/<project>.json")
def get_project_data(project):
    data_file = "data/staff.xlsx"  # Adjust path if needed
    data_df = pd.read_excel(data_file)

    # Filter data for the selected project
    filtered_data = data_df[data_df["Project"] == project]

    # Get unique jobs and months from the filtered data
    unique_jobs = list(filtered_data["Job"].unique())
    months = list(filtered_data["Month"].unique())

    # Create a dictionary to store job counts for each month
    job_counts = {}
    for job in unique_jobs:
        job_counts[job] = [0] * len(months)  # Initialize with zeros
        for index, row in filtered_data.iterrows():
            if row["Job"] == job and row["Month"] in months:
                job_counts[job][months.index(row["Month"])] += 1  # Count occurrences

    # Convert the data into JSON format
    data = {
        "months": months,
        "jobs": job_counts,
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
