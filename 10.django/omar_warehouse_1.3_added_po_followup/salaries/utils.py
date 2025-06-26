from django.db import connection

def get_distinct_project_names():
    with connection.cursor() as cursor:
        cursor.execute("SELECT DISTINCT project FROM salaries")
        results = cursor.fetchall()
    return [row[0] for row in results if row[0]]