from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def home(request):
    return render(request, 'home.html')

@login_required
def indirect_costs_report(request):
    context = {}
    return render(request, 'indirect_costs/report.html', context)