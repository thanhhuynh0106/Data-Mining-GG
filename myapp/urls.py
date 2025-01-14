from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_content, name='home_content'),
    path('home/', views.home, name='home'),

    path('upload/', views.upload_view, name='upload'),
    path('upload_data_csv', views.upload_data_csv, name='upload_data_csv'),

    path('view_data/', views.view_data, name='view_data'),
    path('view_results/', views.view_results, name='view_results'),

    path('calculate-approximation/', views.calculate_approximation, name='calculate_approximation'),
    path('approximation/', views.approximation_view, name='approximation'),
    path('show-data/', views.show_data, name='show_data'),

    path('calculate-reduct/', views.calculate_reduct, name='calculate_reduct'),
    path('reduct/', views.reduct_view, name='reduct'),
    path('show-data-reduct/', views.show_data_reduct, name='show_data_reduct'),

    path('apriori/', views.apriori_view, name='apriori'),
    path('calculate-apriori/', views.calculate_apriori, name='calculate_apriori'),
    path('show-data-apriori/', views.show_data_apriori, name='show_data_apriori'),

    path('decisiontree/', views.decision_tree_view, name='decisiontree'),
    path('show-data-decisiontree/', views.show_data_decision_tree, name='show_data_decisiontree'),
    path('calculate-decisiontree/', views.calculate_decision_tree, name='calculate_decisiontree'),

    path('naivebayes/', views.naivebayes_view, name='naivebayes'),
    path('show_data_naivebayes/', views.show_data_naivebayes, name='show_data_naivebayes'),
    path('calculate_naivebayes/', views.calculate_naivebayes, name='calculate_naivebayes'),
    path('calculate_naivebayes_smoothing/', views.calculate_naivebayes_smoothing, name='calculate_naivebayes_smoothing'),

    path('kmeans/', views.kmeans_view, name='kmeans'),
    path('show_data_kmeans/', views.show_data_kmeans, name='show_data_kmeans'),
    path('calculate_kmeans/', views.calculate_kmeans, name='calculate_kmeans'),
    
    path('kohonen/', views.kohonen_view, name='kohonen'),
    path('show_data_kohonen/', views.show_data_kohonen, name='show_data_kohonen'),
    path('calculate_kohonen/', views.calculate_kohonen, name='calculate_kohonen'),
]   