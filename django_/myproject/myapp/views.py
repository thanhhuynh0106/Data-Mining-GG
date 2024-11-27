from django.shortcuts import render


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def home(request):
    return render(request, 'home.html')

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
    return render(request, 'apriori.html', context)


# Upload file

def upload_data(request):
    preview_data = None  # Dữ liệu xem trước
    if request.method == "POST":
        uploaded_file = request.FILES.get('file')
        if uploaded_file:
            # Đọc file Excel/CSV để xem trước dữ liệu
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    raise ValueError("Chỉ hỗ trợ file CSV hoặc Excel.")
                
                # Lấy 5 dòng đầu tiên làm dữ liệu xem trước
                preview_data = df.head().to_html(classes='table table-striped', index=False)
            except Exception as e:
                preview_data = f"Lỗi: {str(e)}"
    
    return render(request, 'upload_data.html', {'preview_data': preview_data})
