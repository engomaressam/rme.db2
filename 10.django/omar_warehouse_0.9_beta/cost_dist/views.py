from django.shortcuts import render
from django.http import HttpResponse
from django.db import connection
from .forms import CostDistForm
from .utils import get_distinct_project_names 
import openpyxl
from openpyxl.styles import Font

def cost_dist_report(request):
    if request.method == 'POST':
        form = CostDistForm(request.POST)
        if form.is_valid():
            project_identifier_type = form.cleaned_data['project_identifier_type']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            show_details = form.cleaned_data['show_details']

            results = []

            with connection.cursor() as cursor:
                if project_identifier_type == 'project_no':
                    project_identifier = form.cleaned_data['project_identifier']
                    if show_details:
                        # Detailed view for project number
                        query = f"""
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
                        params = [project_identifier]
                        if start_date and end_date:
                            params.extend([start_date, end_date])

                        try:
                            cursor.execute(query, params)
                            results = cursor.fetchall()
                        except Exception as e:
                            print(f"Error executing query: {e}")
                            results = [] 

                    else:
                        # Total amount for project number
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
                        params = [project_identifier]
                        if start_date and end_date:
                            params.extend([start_date, end_date])

                        try:
                            cursor.execute(query, params)
                            results = cursor.fetchall()
                        except Exception as e:
                            print(f"Error executing query: {e}")
                            results = [] 

                else:  # project_identifier_type == 'project_name'
                    project_name = form.cleaned_data['project_name']
                    if show_details:
                        # Detailed view for project name
                        query = f"""
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
                        params = [project_name] 
                        if start_date and end_date:
                            params.extend([start_date, end_date])

                        try:
                            cursor.execute(query, params)
                            results = cursor.fetchall()
                        except Exception as e:
                            print(f"Error executing query: {e}")
                            results = [] 

                    else:
                        # Total amount for project name
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
                        params = [project_name]
                        if start_date and end_date:
                            params.extend([start_date, end_date])

                        try:
                            cursor.execute(query, params)
                            results = cursor.fetchall()
                        except Exception as e:
                            print(f"Error executing query: {e}")
                            results = [] 

            # Calculate total amount regardless of show_details
            if results:  
                total_amount = sum(row[2] for row in results) 
                formatted_total_amount = "{:,.2f}".format(total_amount)
            else:
                formatted_total_amount = 0 

            # Excel export (only when export_excel button is clicked)
            if request.POST.get('export_excel'):
                response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                response['Content-Disposition'] = 'attachment; filename=cost_dist_report.xlsx'
                workbook = openpyxl.Workbook()
                worksheet = workbook.active
                worksheet.append(['Line Description', 'Unit', 'Quantity', 'Amount']) 

                # Fetch detailed results for export (if not already fetched)
                if not show_details:
                    with connection.cursor() as cursor:
                        if project_identifier_type == 'project_no':
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
                            params_details = [project_identifier] 
                            if start_date and end_date:
                                params_details.extend([start_date, end_date])

                        else:  # project_identifier_type == 'project_name'
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
                            params_details = [project_name] 
                            if start_date and end_date:
                                params_details.extend([start_date, end_date])

                        cursor.execute(query_details, params_details)
                        results = cursor.fetchall()

                for row in results:
                    worksheet.append(list(row))

                # Apply bold style to the header row
                for cell in worksheet[1]:
                    cell.font = Font(bold=True)

                workbook.save(response)
                return response

            context = {'results': results, 'total_amount': formatted_total_amount} 

            return render(request, 'cost_dist/report.html', context)

    else:
        form = CostDistForm()

    return render(request, 'cost_dist/report.html', {'form': form})