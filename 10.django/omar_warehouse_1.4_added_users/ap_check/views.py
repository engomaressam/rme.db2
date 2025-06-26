from django.shortcuts import render
from django.db import connection
from .forms import ApCheckForm  # We'll create this form later
from .utils import get_distinct_project_names
from django.contrib.auth.decorators import login_required

@login_required
def ap_check_report(request):
    if request.method == 'POST':
        form = ApCheckForm(request.POST)
        if form.is_valid():
            project_name = form.cleaned_data['project_name']
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            show_details = form.cleaned_data['show_details']

            # Construct and execute the SQL query 
            with connection.cursor() as cursor:
                query = f"""
                    SELECT 
                        project_name, 
                        SUM(equiv) AS total_cash_out
                    FROM 
                        ap_check
                    WHERE 
                        LOWER(project_name) = LOWER(%s) 
                        {f'AND check_date BETWEEN %s AND %s' if start_date and end_date else ''} 
                    GROUP BY 
                        project_name;
                """

                # Adjust params based on whether dates are provided
                if start_date and end_date:
                    params = [project_name, start_date, end_date]
                else:
                    params = [project_name]

                cursor.execute(query, params)
                results = cursor.fetchall()

            if show_details:
                # Fetch detailed results if needed (you'll need to adjust the query for this)
                context = {'results': results} 
            else:
                total_cash_out = sum(row[1] for row in results)  # Assuming 'total_cash_out' is at index 1
                formatted_total_cash_out = "{:,.2f}".format(total_cash_out)
                context = {'total_cash_out': formatted_total_cash_out}

            return render(request, 'ap_check/report.html', context)

    else:
        form = ApCheckForm()

    return render(request, 'ap_check/report.html', {'form': form})