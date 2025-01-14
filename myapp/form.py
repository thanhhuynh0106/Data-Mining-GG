# forms.py
from django import forms

class DataUploadForm(forms.Form):
    ALGORITHM_CHOICES = [
        ('apriori', 'Apriori'),
        ('kmeans', 'KMeans'),
        ('kohonen', 'Kohonen'),
    ]
    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES, label='Select Algorithm')
    data_file = forms.FileField(label='Upload Data File')