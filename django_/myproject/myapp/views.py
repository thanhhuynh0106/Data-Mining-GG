from django.shortcuts import render

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def home(request):
    return render(request, 'myapp/home.html')

def display_csv(request):
    import pandas as pd
    df = pd.read_csv('myapp/datasets/Groceries_dataset.csv')
    html_table = df.to_html()
    return render(request, 'myapp/display_csv.html', {'html_table': html_table})

def apriori_view(request):
    import pandas as pd
    from mlxtend.frequent_patterns import apriori, association_rules
    df = pd.read_csv('myapp/datasets/Groceries_dataset.csv')
    basket = (df.groupby(['Member_number', 'itemDescription'])['Date'].count().unstack().reset_index().fillna(0).set_index('Member_number'))

    def encode_units(x):
        if x < 1:
            return 0
        if x >= 1:
            return 1

    basket = basket.applymap(encode_units)
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1, num_itemsets=10)
    context = {'rules': rules.to_html()}
    return render(request, 'myapp/apriori.html', context)

