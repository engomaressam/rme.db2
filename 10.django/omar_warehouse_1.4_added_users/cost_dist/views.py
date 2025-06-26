from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection
from datetime import datetime, timedelta
from .forms import CostDistForm
from .utils import get_distinct_project_names 
import openpyxl
from openpyxl.styles import Font
from django.contrib.auth.decorators import login_required

@login_required
def cost_dist_report(request):
    if request.method == 'POST':
        form = CostDistForm(request.POST)
        if form.is_valid():
            project_identifier_type = form.cleaned_data['project_identifier_type']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            show_details = form.cleaned_data['show_details']

            with connection.cursor() as cursor:
                if project_identifier_type == 'project_no':
                    project_identifier = form.cleaned_data['project_identifier']
                    # Query for total amount 
                    query_total = f"""
                        SELECT 
                            project_no, 
                            project_name, 
                            SUM(amount) AS total_cost
                        FROM 
                            cost_dist
                        WHERE 
                            project_no = %s 
                            {f'AND gl_date BETWEEN %s AND %s' if start_date and end_date else ''} 
                        GROUP BY 
                            project_no, project_name;
                    """

                    # Query for detailed data (for Excel export)
                    query_details = f"""
                        SELECT 
                            line_desc, 
                            unit, 
                            qty, 
                            amount
                        FROM 
                            cost_dist
                        WHERE 
                            project_no = %s 
                            {f'AND gl_date BETWEEN %s AND %s' if start_date and end_date else ''} 
                    """ 

                else:  # project_identifier_type == 'project_name'
                    project_name = form.cleaned_data['project_name']
                    # Query for total amount
                    query_total = f"""
                        SELECT 
                            project_no, 
                            project_name, 
                            SUM(amount) AS total_cost
                        FROM 
                            cost_dist
                        WHERE 
                            LOWER(project_name) = LOWER(%s) 
                            {f'AND gl_date BETWEEN %s AND %s' if start_date and end_date else ''} 
                        GROUP BY 
                            project_no, project_name;
                    """

                    # Query for detailed data (for Excel export)
                    query_details = f"""
                        SELECT 
                            line_desc, 
                            unit, 
                            qty, 
                            amount
                        FROM 
                            cost_dist
                        WHERE 
                            LOWER(project_name) = LOWER(%s) 
                            {f'AND gl_date BETWEEN %s AND %s' if start_date and end_date else ''} 
                    """

                # Adjust params based on whether dates are provided
                if start_date and end_date:
                    params = [project_identifier if project_identifier_type == 'project_no' else project_name, start_date, end_date]
                else:
                    params = [project_identifier if project_identifier_type == 'project_no' else project_name]

                try:
                    # Always execute the query for total amount 
                    cursor.execute(query_total, params)
                    results_total = cursor.fetchall()

                    # Calculate and format the total amount
                    total_amount = sum(row[2] for row in results_total) if results_total else 0
                    formatted_total_amount = "{:,.2f}".format(total_amount)

                    # Excel export or detailed view
                    if show_details or request.POST.get('export_excel'): 
                        cursor.execute(query_details, params)  # Execute the detailed query only if needed
                        results_details = cursor.fetchall()

                        if request.POST.get('export_excel'):
                            response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                            response['Content-Disposition'] = 'attachment; filename=cost_dist_report.xlsx'
                            workbook = openpyxl.Workbook()
                            worksheet = workbook.active
                            worksheet.append(['Line Description', 'Unit', 'Quantity', 'Amount']) 
                            for row in results_details:
                                worksheet.append(list(row))

                            # Apply bold style to the header row
                            for cell in worksheet[1]:
                                cell.font = Font(bold=True)

                            workbook.save(response)
                            return response
                        else: 
                            context = {'results': results_details, 'total_amount': formatted_total_amount}
                    else:
                        context = {'total_amount': formatted_total_amount}

                    return render(request, 'cost_dist/report.html', context)

                except Exception as e:
                    print(f"Error executing query: {e}")
                    return HttpResponse("An error occurred while processing your request.")

    else:
        form = CostDistForm()

    return render(request, 'cost_dist/report.html', {'form': form})

























@login_required
def cost_dist_chart(request):
    if request.method == 'POST':
        # Get project identifier from the form data 
        project_identifier_type = request.POST.get('project_identifier_type')
        if project_identifier_type == 'project_no':
            project_identifier = request.POST.get('project_identifier')
        else:
            project_identifier = request.POST.get('project_name')

        # Calculate the date range for the last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=6*30)

        with connection.cursor() as cursor:
            if project_identifier_type == 'project_no':
                # Query for the last 6 months' costs by project number
                query = """
                    SELECT 
                        MONTH(gl_date) as month, 
                        YEAR(gl_date) as year,
                        SUM(amount) as total_cost
                    FROM 
                        cost_dist
                    WHERE 
                        project_no = %s 
                        AND gl_date BETWEEN %s AND %s
                    GROUP BY 
                        MONTH(gl_date), YEAR(gl_date)
                    ORDER BY 
                        YEAR(gl_date), MONTH(gl_date);
                """
                params = [project_identifier, start_date, end_date]
            else:
                # Query for the last 6 months' costs by project name
                query = """
                    SELECT 
                        MONTH(gl_date) as month, 
                        YEAR(gl_date) as year,
                        SUM(amount) as total_cost
                    FROM 
                        cost_dist
                    WHERE 
                        LOWER(project_name) = LOWER(%s) 
                        AND gl_date BETWEEN %s AND %s
                    GROUP BY 
                        MONTH(gl_date), YEAR(gl_date)
                    ORDER BY 
                        YEAR(gl_date), MONTH(gl_date);
                """
                params = [project_identifier, start_date, end_date]

            try:
                cursor.execute(query, params)
                results = cursor.fetchall()

                # Prepare data for the chart (format month labels)
                chart_data = {
                    'labels': [datetime(year=row[1], month=row[0], day=1).strftime('%b-%y') for row in results], 
                    'data': [row[2] for row in results]
                }

            except Exception as e:
                print(f"Error executing query: {e}")
                chart_data = {'labels': [], 'data': []} 

            context = {
                'project_identifier': project_identifier,
                'chart_data': chart_data
            }

            return render(request, 'cost_dist/chart.html', context)

    else:  # Handle GET requests (initial page load)
        # You might want to redirect to the main report page or display an error message here
        return render(request, 'cost_dist/report.html', {'form': CostDistForm()})