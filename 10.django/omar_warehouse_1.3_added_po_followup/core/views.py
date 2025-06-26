from django.shortcuts import render

def home(request):
    return render(request, 'home.html')

def indirect_costs_report(request):
    context = {}
    return render(request, 'indirect_costs/report.html', context)