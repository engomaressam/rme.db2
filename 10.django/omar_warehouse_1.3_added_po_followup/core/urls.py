from django.contrib import admin
from django.urls import path
from cost_dist import views as cost_dist_views
from ap_check import views as ap_check_views
from salaries import views as salaries_views
from po_followup import views as po_followup_views
# from wages import views as wages_views
from core import views 

urlpatterns = [

    path('', views.home, name='home'),
    path('cost_dist_report/', cost_dist_views.cost_dist_report, name='cost_dist_report'),
    path('ap_check_report/', ap_check_views.ap_check_report, name='ap_check_report'),
    path('indirect_costs_report/', views.indirect_costs_report, name='indirect_costs_report'),
    path('indirect_costs_report/salaries/', salaries_views.salaries_report, name='indirect_costs_salaries'),
    # path('indirect_costs_report/wages/', views.indirect_costs_wages, name='indirect_costs_wages'),
    # path('indirect_costs_report/rented_cars/', views.indirect_costs_rented_cars, name='indirect_costs_rented_cars'),
    path('cost_dist_report/chart/', cost_dist_views.cost_dist_chart, name='cost_dist_chart'),
    path('po_followup_report/', po_followup_views.po_followup_report, name='po_followup_report'),  # Add this line
    path('po_followup_report/get_matching_descriptions/', po_followup_views.get_matching_descriptions, name='get_matching_descriptions'),
]
