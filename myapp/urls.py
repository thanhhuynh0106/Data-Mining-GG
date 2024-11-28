from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_content, name='home_content'),
    path('apriori/', views.apriori_view, name='apriori'),
    path('upload_data/', views.upload_data, name='upload_data'),
    path('view_data/', views.view_data, name='view_data'),
    path('view_results/', views.view_results, name='view_results'),
    path('calculate-approximation/', views.calculate_approximation, name='calculate_approximation'),
    path('approximation/', views.approximation_view, name='approximation'),
    path('show-data/', views.show_data, name='show_data'),
]   