from django.shortcuts import render
from django.db import connection
from .forms import IndirectCostForm
from .utils import get_distinct_project_names

def salaries_report(request):
    if request.method == 'POST':
        form = IndirectCostForm(request.POST) 
        if form.is_valid():
            project_name = form.cleaned_data['project_name']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']

            with connection.cursor() as cursor:
                # Query for total salaries
                query_total = f"""
                    SELECT 
                        SUM(amount) AS total_salaries
                    FROM 
                        salaries
                    WHERE 
                        project = %s 
                        {f'AND month BETWEEN %s AND %s' if start_date and end_date else ''} 
                """

                # Query for last month's salaries
                query_last_month = """
                    SELECT TOP 1 
                        month, 
                        SUM(amount) AS last_month_salaries
                    FROM 
                        salaries
                    WHERE 
                        project = %s
                        AND amount IS NOT NULL  
                    GROUP BY 
                        month
                    ORDER BY 
                        month DESC;
                """

                params = [project_name]
                if start_date and end_date:
                    params.extend([start_date, end_date])

                try:
                    # Execute queries
                    cursor.execute(query_total, params)
                    total_salaries_result = cursor.fetchone()
                    cursor.execute(query_last_month, [project_name]) 
                    last_month_result = cursor.fetchone()

                except Exception as e:
                    print(f"Error executing query: {e}")
                    total_salaries_result = None
                    last_month_result = None

            # Process and format results
            total_salaries = total_salaries_result[0] if total_salaries_result else 0
            formatted_total_salaries = f"{total_salaries:,.2f} EGP"

            if last_month_result:
                last_month = last_month_result[0].strftime('%b-%y')
                last_month_salaries = last_month_result[1]
                formatted_last_month_salaries = f"{last_month_salaries:,.2f} EGP"
                last_month_info = f"Last Month ({last_month}) Total Salaries = {formatted_last_month_salaries}"
            else:
                last_month_info = "No data available for the last month."

            context = {
                'project_name': project_name,
                'total_salaries': formatted_total_salaries,
                'last_month_info': last_month_info,
            }

            return render(request, 'indirect_costs/salaries_report.html', context) # Update the template path
    else:
        form = IndirectCostForm()

    return render(request, 'indirect_costs/salaries_report.html', {'form': form})