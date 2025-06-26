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
        .container { max-width: 900px; margin: 40px auto; }
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
        <span class="fs-3 fw-bold ms-2">Sector Financial Summary <span class="fs-6 text-secondary">({{ date_range }})</span></span>
    </div>
    <div class="card p-4">
        <form method="post" class="row g-3 align-items-end">
            <div class="col-md-5">
                <label for="sector" class="form-label">Select Sector</label>
                <select name="sector" id="sector" class="form-select">
                    {% for sector in sectors %}
                        <option value="{{ sector }}" {% if sector == selected_sector %}selected{% endif %}>{{ sector }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="col-md-3">
                <label for="from_date" class="form-label">From Date</label>
                <input type="date" name="from_date" id="from_date" class="form-control" value="{{ from_date }}">
            </div>
            <div class="col-md-3">
                <label for="to_date" class="form-label">To Date</label>
                <input type="date" name="to_date" id="to_date" class="form-control" value="{{ to_date }}">
            </div>
            <div class="col-md-1 d-grid">
                <button type="submit" class="btn btn-success">Show</button>
            </div>
        </form>
        {% if sector_totals %}
            <div class="mt-4">
                <h4>Totals for <span class="text-primary">{{ selected_sector }}</span>:</h4>
                <div class="row mb-3">
                    <div class="col-md-4"><span class="fw-bold">Cost:</span> <span class="total">{{ '{:,}'.format(sector_totals['COST']) }}</span></div>
                    <div class="col-md-4"><span class="fw-bold">Cash Out:</span> <span class="total">{{ '{:,}'.format(sector_totals['CASH_OUT']) }}</span></div>
                    <div class="col-md-4"><span class="fw-bold">Cash In:</span> <span class="total">{{ '{:,}'.format(sector_totals['CASH_IN']) }}</span></div>
                </div>
                <h5 class="mb-2">Breakdown by Project</h5>
                <div class="table-responsive">
                    <table class="table table-striped table-bordered">
                        <thead class="table-light">
                            <tr>
                                <th>Project Number</th>
                                <th>Project Name</th>
                                <th>Cost</th>
                                <th>Cash Out</th>
                                <th>Cash In</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for row in breakdown %}
                            <tr>
                                <td>{{ row['PROJECT_NUM'] }}</td>
                                <td>{{ row['PROJECT_NAME'] }}</td>
                                <td>{{ '{:,}'.format(row['COST']) }}</td>
                                <td>{{ '{:,}'.format(row['CASH_OUT']) }}</td>
                                <td>{{ '{:,}'.format(row['CASH_IN']) }}</td>
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

def get_all_sums_for_sector(sector, from_date=None, to_date=None):
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
    sector_projects = projects_df[projects_df['SECTOR'] == sector][['PROJECT_NUM', 'PROJECT_NAME']]
    if sector_projects.empty:
        return {'COST': 0, 'CASH_OUT': 0, 'CASH_IN': 0}, []
    breakdown = []
    sector_totals = {'COST': 0, 'CASH_OUT': 0, 'CASH_IN': 0}
    try:
        cnx = mysql.connect(**db_config)
        cursor = cnx.cursor()
        for _, row in sector_projects.iterrows():
            project_num = row['PROJECT_NUM']
            project_name = row['PROJECT_NAME']
            project_sums = {}
            for metric, m in metric_map.items():
                query = f"SELECT SUM({m['amount_col']}) FROM {m['table']} WHERE {m['project_col']} = %s"
                params = [project_num]
                if from_date:
                    query += f" AND {m['date_col']} >= %s"
                    params.append(from_date)
                if to_date:
                    query += f" AND {m['date_col']} <= %s"
                    params.append(to_date)
                cursor.execute(query, tuple(params))
                result = cursor.fetchone()
                amount = float(result[0]) if result[0] is not None else 0.0
                project_sums[metric] = int(amount)
                sector_totals[metric] += int(amount)
            breakdown.append({'PROJECT_NUM': project_num, 'PROJECT_NAME': project_name, **project_sums})
        cursor.close()
        cnx.close()
    except Error as e:
        print(f"Error connecting to database: {e}")
    return sector_totals, breakdown

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_sector = dropdown_sectors[0]
    sector_totals = None
    breakdown = []
    from_date = ''
    to_date = ''
    date_range = 'All Dates'
    if request.method == 'POST':
        selected_sector = request.form.get('sector', selected_sector)
        from_date = request.form.get('from_date', '').strip()
        to_date = request.form.get('to_date', '').strip()
        if from_date and to_date:
            date_range = f"{from_date} to {to_date}"
        elif from_date:
            date_range = f"From {from_date}"
        elif to_date:
            date_range = f"Up to {to_date}"
        sector_totals, breakdown = get_all_sums_for_sector(selected_sector, from_date or None, to_date or None)
    return render_template_string(TEMPLATE,
                                  sectors=dropdown_sectors,
                                  selected_sector=selected_sector,
                                  sector_totals=sector_totals,
                                  breakdown=breakdown,
                                  from_date=from_date,
                                  to_date=to_date,
                                  date_range=date_range,
                                  year=datetime.now().year)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5055) 