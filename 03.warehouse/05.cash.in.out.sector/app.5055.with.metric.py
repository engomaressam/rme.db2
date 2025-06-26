from flask import Flask, render_template_string, request
import pandas as pd
import os
from datetime import datetime
import mysql.connector as mysql
from mysql.connector import Error

app = Flask(__name__)

# Path to the CSV file
dir_path = os.path.dirname(os.path.abspath(__file__))
projects_csv = os.path.join(dir_path, 'unique_projects_with_sectors.csv')

# MySQL connection settings
db_config = {
    'host': '10.10.11.242',
    'user': 'omar2',
    'password': 'Omar_54321',
    'database': 'RME_TEST'
}

# Load project/sector mapping from CSV
projects_df = pd.read_csv(projects_csv)
dropdown_sectors = sorted(projects_df['SECTOR'].dropna().unique())
dropdown_metrics = ['COST', 'CASH_OUT', 'CASH_IN']

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sector Financial Summary</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f8f9fa; }
        .container { max-width: 800px; margin: 40px auto; }
        .card { box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
        .form-select, .form-control, .btn { min-width: 150px; }
        .total { font-weight: bold; color: #198754; font-size: 1.3em; }
        .footer { text-align: center; color: #888; margin-top: 40px; font-size: 0.95em; }
        .table-responsive { margin-top: 20px; }
    </style>
</head>
<body>
<div class="container">
    <div class="text-center mb-4">
        <img src="https://img.icons8.com/ios-filled/50/198754/money-bag.png" alt="icon" style="height:40px;vertical-align:middle;">
        <span class="fs-3 fw-bold ms-2">Sector Financial Summary <span class="fs-6 text-secondary">(up to 30-Mar-2025)</span></span>
    </div>
    <div class="card p-4">
        <form method="post" class="row g-3 align-items-end">
            <div class="col-md-6">
                <label for="sector" class="form-label">Select Sector</label>
                <select name="sector" id="sector" class="form-select">
                    {% for sector in sectors %}
                        <option value="{{ sector }}" {% if sector == selected_sector %}selected{% endif %}>{{ sector }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="col-md-4">
                <label for="metric" class="form-label">Select Metric</label>
                <select name="metric" id="metric" class="form-select">
                    {% for metric in metrics %}
                        <option value="{{ metric }}" {% if metric == selected_metric %}selected{% endif %}>{{ metric.replace('_', ' ').title() }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="col-md-2 d-grid">
                <button type="submit" class="btn btn-success">Show</button>
            </div>
        </form>
        {% if total is not none %}
            <div class="mt-4">
                <h4>Total <span class="text-capitalize">{{ selected_metric.replace('_', ' ').title() }}</span> for <span class="text-primary">{{ selected_sector }}</span>:</h4>
                <div class="total mb-3">{{ '{:,.2f}'.format(total) }}</div>
                <h5 class="mb-2">Breakdown by Project</h5>
                <div class="table-responsive">
                    <table class="table table-striped table-bordered">
                        <thead class="table-light">
                            <tr>
                                <th>Project Number</th>
                                <th>Project Name</th>
                                <th>{{ selected_metric.replace('_', ' ').title() }}</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for row in breakdown %}
                            <tr>
                                <td>{{ row['PROJECT_NUM'] }}</td>
                                <td>{{ row['PROJECT_NAME'] }}</td>
                                <td>{{ '{:,.2f}'.format(row[selected_metric]) }}</td>
                            </tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        {% endif %}
    </div>
    <div class="footer">
        &copy; {{ year }} RME Data Science | Powered by Flask & Bootstrap
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

# Helper to get live sums from MySQL
def get_sums_for_sector(sector, metric):
    # Map metric to table/column/date
    metric_map = {
        'COST': {
            'table': 'RME_Projects_Cost_Dist_Line_Report',
            'amount_col': 'AMOUNT',
            'project_col': 'PROJECT_NUM',
            'date_col': 'GL_DATE'
        },
        'CASH_OUT': {
            'table': 'RME_ap_check_payments_Report',
            'amount_col': 'EQUIV',
            'project_col': 'PROJECT_NUMBER',
            'date_col': 'CHECK_DATE'
        },
        'CASH_IN': {
            'table': 'SWD_Collection_Report',
            'amount_col': 'FUNC_AMOUNT',
            'project_col': 'PROJECT_NUM',
            'date_col': 'RECEIPT_DATE'
        }
    }
    m = metric_map[metric]
    # Get all projects in this sector
    sector_projects = projects_df[projects_df['SECTOR'] == sector][['PROJECT_NUM', 'PROJECT_NAME']]
    if sector_projects.empty:
        return 0, []
    # Prepare results
    breakdown = []
    total = 0
    try:
        cnx = mysql.connect(**db_config)
        cursor = cnx.cursor()
        for _, row in sector_projects.iterrows():
            project_num = row['PROJECT_NUM']
            project_name = row['PROJECT_NAME']
            # Query sum for this project
            query = f"""
                SELECT SUM({m['amount_col']}) FROM {m['table']}
                WHERE {m['project_col']} = %s AND {m['date_col']} <= '2025-03-30'
            """
            cursor.execute(query, (project_num,))
            result = cursor.fetchone()
            amount = float(result[0]) if result[0] is not None else 0.0
            breakdown.append({'PROJECT_NUM': project_num, 'PROJECT_NAME': project_name, metric: amount})
            total += amount
        cursor.close()
        cnx.close()
    except Error as e:
        print(f"Error connecting to database: {e}")
    return total, breakdown

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_sector = dropdown_sectors[0]
    selected_metric = dropdown_metrics[0]
    total = None
    breakdown = []
    if request.method == 'POST':
        selected_sector = request.form.get('sector', selected_sector)
        selected_metric = request.form.get('metric', selected_metric)
        total, breakdown = get_sums_for_sector(selected_sector, selected_metric)
    return render_template_string(TEMPLATE,
                                  sectors=dropdown_sectors,
                                  metrics=dropdown_metrics,
                                  selected_sector=selected_sector,
                                  selected_metric=selected_metric,
                                  total=total,
                                  breakdown=breakdown,
                                  year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5055) 