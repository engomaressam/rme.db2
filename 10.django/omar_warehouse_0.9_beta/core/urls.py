from django.contrib import admin
from django.urls import path
from cost_dist import views as cost_dist_views
from ap_check import views as ap_check_views
from core import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('cost_dist_report/', cost_dist_views.cost_dist_report, name='cost_dist_report'),
    path('ap_check_report/', ap_check_views.ap_check_report, name='ap_check_report'),
    path('', views.home, name='home'),
]