from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('display_csv/', views.display_csv, name='display_csv'),
    path('apriori/', views.apriori_view, name='apriori'),
    path('upload-data/', views.upload_data, name='upload_data'),
]   