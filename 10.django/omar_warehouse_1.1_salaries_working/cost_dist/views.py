from django.shortcuts import render
from django.db import connection
from .forms import CostDistForm
from .utils import get_distinct_project_names

def cost_dist_report(request):
    if request.method == 'POST':
        form = CostDistForm(request.POST)
        if form.is_valid():
            project_identifier_type = form.cleaned_data['project_identifier_type']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            show_details = form.cleaned_data['show_details']

            try:
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

                    cursor.execute(query, params)
                    results = cursor.fetchall()

            except Exception as e:
                # Handle any database errors gracefully
                print(f"Error executing query: {e}")
                results = []  # Set results to an empty list in case of an error

            if show_details:
                context = {'results': results}
            else:
                total_amount = sum(row[2] for row in results) if results else 0 

                # Format the total_amount with commas
                formatted_total_amount = "{:,.2f}".format(total_amount)

                context = {'total_amount': formatted_total_amount}

            return render(request, 'cost_dist/report.html', context)

    else:
        form = CostDistForm()

    return render(request, 'cost_dist/report.html', {'form': form})