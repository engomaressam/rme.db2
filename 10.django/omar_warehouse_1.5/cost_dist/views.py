from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection
from datetime import datetime, timedelta
from .forms import CostDistForm
from .utils import get_distinct_project_names 
import openpyxl
from openpyxl.styles import Font
from django.contrib.auth.decorators import login_required
from datetime import datetime

@login_required
def cost_dist_report(request):
    if request.method == 'POST':
        form = CostDistForm(request.POST)
        if form.is_valid():
            project_identifier_type = form.cleaned_data['project_identifier_type']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']

            with connection.cursor() as cursor:
                if project_identifier_type == 'project_no':
                    project_identifier = form.cleaned_data['project_identifier']
                    # Query for total amount for project number
                    query = f"""
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

                else:  # project_identifier_type == 'project_name'
                    project_name = form.cleaned_data['project_name']
                    
                    # Query for total amount for project name
                    query = f"""
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

                # Adjust params based on whether dates are provided
                if start_date and end_date:
                    params = [project_identifier if project_identifier_type == 'project_no' else project_name, start_date, end_date]
                else:
                    params = [project_identifier if project_identifier_type == 'project_no' else project_name]

                try:
                    cursor.execute(query, params)
                    results = cursor.fetchall()
                    print("Query executed successfully!")
                    print("Number of results:", len(results))
                    print("First 5 results:", results[:5])

                except Exception as e:
                    print(f"Error executing query: {e}")
                    results = [] 

            # Calculate total amount
            if results:  
                total_amount = sum(row[2] for row in results)
                formatted_total_amount = "{:,.2f}".format(total_amount)
            else:
                formatted_total_amount = 0 

            context = {'total_amount': formatted_total_amount} 

            return render(request, 'cost_dist/report.html', context)

    else:
        form = CostDistForm()
        # Fetch distinct project names and populate the form's choices
        project_names = get_distinct_project_names()
        form.fields['project_name'].choices = [('', '--- Select a Project Name ---')] + [(name, name) for name in project_names]

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
    

















@login_required
def export_to_excel(request):
    if request.method == 'GET':  
        project_identifier_type = request.GET.get('project_identifier_type')
        project_identifier = request.GET.get('project_identifier')
        start_date_str = request.GET.get('start_date')
        end_date_str = request.GET.get('end_date')

        print(f"Received parameters: project_identifier_type={project_identifier_type}, project_identifier={project_identifier}, start_date={start_date_str}, end_date={end_date_str}") 

        # Check if project_identifier_type and project_identifier are provided
        if not project_identifier_type or not project_identifier:
            return HttpResponse("Please select a project identifier (number or name) before exporting.")

        # Convert start_date and end_date strings to datetime objects only if they are not 'None'
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d') if start_date_str and start_date_str != 'None' else None
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d') if end_date_str and end_date_str != 'None' else None
        except ValueError:
            return HttpResponse("Invalid date format. Please use YYYY-MM-DD.")

        with connection.cursor() as cursor:
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
                    {f'project_no = %s' if project_identifier_type == 'project_no' else 'LOWER(project_name) = LOWER(%s)'} 
                    {f'AND gl_date BETWEEN %s AND %s' if start_date and end_date else ''} 
            """ 

            # Adjust params based on whether dates are provided
            if start_date and end_date:
                params = [project_identifier, start_date, end_date]
            else:
                params = [project_identifier]

            try:
                cursor.execute(query_details, params)  
                results = cursor.fetchall()

                print("Excel Export Query:", query_details) 
                print("Excel Export Params:", params) 
                print("Number of results for Excel export:", len(results))
                print("First 5 results for Excel export:", results[:5])

                response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                response['Content-Disposition'] = 'attachment; filename=cost_dist_report.xlsx'
                workbook = openpyxl.Workbook()
                worksheet = workbook.active
                worksheet.append(['Line Description', 'Unit', 'Quantity', 'Amount']) 
                for row in results:
                    worksheet.append(list(row))

                # Apply bold style to the header row
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)

                workbook.save(response)
                return response 

            except Exception as e:
                print(f"Error executing query or generating Excel: {e}")
                return HttpResponse("An error occurred while processing your request.")

    # You might want to handle other HTTP methods (e.g., GET) or invalid requests here
    return HttpResponse("Invalid request.")