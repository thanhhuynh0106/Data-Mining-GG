from django.http import HttpResponse, JsonResponse
from django.shortcuts import render


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def home_content(request):
    return render(request, 'myapp/home.html')


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


def upload_data(request):
    if request.method == 'POST':
        file = request.FILES['file']
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
        request.session['data'] = data.to_json()
        return JsonResponse({'success': True})
    return JsonResponse({'success': False})


def view_data(request):
    data_json = request.session.get('data')
    if data_json:
        data = pd.read_json(data_json)
        html = data.to_html()
        return HttpResponse(html)
    return HttpResponse('No data available')

def view_results(request):
    # Logic to calculate results based on the uploaded data
    results = "Kết quả tính toán"
    return HttpResponse(results)


"""   APPROXIMATION    """

def show_data(request):
    try:
        # Đọc dữ liệu từ file CSV
        data = pd.read_csv(file_path1)
        # Chuyển dữ liệu thành HTML table
        data_html = data.to_html(index=False, classes="table table-bordered", border=0)
        return HttpResponse(data_html)
    except Exception as e:
        return HttpResponse(f"<p style='color: red;'>Error loading data: {e}</p>")

def approximation_view(request):
    return render(request, 'myapp/approximation.html')

file_path1 = 'myapp/datasets/reduct.csv'  # Đảm bảo file này nằm cùng thư mục hoặc chỉ định đúng đường dẫn

def calculate_approximation(request):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(file_path1)

    # Hàm tính quan hệ bất khả phân biệt
    def indiscernibility_relation(data, attributes):
        equivalence_classes = {}
        for idx, row in data.iterrows():
            key = tuple(row[attr] for attr in attributes)  # Lấy giá trị của các thuộc tính
            if key not in equivalence_classes:
                equivalence_classes[key] = []
            equivalence_classes[key].append(idx)
        return equivalence_classes

    # Tính quan hệ bất khả phân biệt cho 'Age' và 'Sessions'
    eq_classes = indiscernibility_relation(data, ['Age', 'Sessions'])

    # Hàm tính xấp xỉ dưới
    def lower_approximation(eq_classes, target_set):
        lower = []
        for eq_class in eq_classes.values():
            if set(eq_class).issubset(target_set):  # Toàn bộ lớp tương đương nằm trong tập mục tiêu
                lower.extend(eq_class)
        return set(lower)

    # Hàm tính xấp xỉ trên
    def upper_approximation(eq_classes, target_set):
        upper = []
        for eq_class in eq_classes.values():
            if set(eq_class).intersection(target_set):  # Có giao với tập mục tiêu
                upper.extend(eq_class)
        return set(upper)

    # Tập hợp các đối tượng có nhãn 'Yes'
    target_set = set(data[data['Pass'] == 'Yes'].index)

    # Tính xấp xỉ dưới và trên
    lower = lower_approximation(eq_classes, target_set)
    upper = upper_approximation(eq_classes, target_set)

    # Trả về kết quả dưới dạng HTML
    return HttpResponse(f"""
    
        <h2>Kết quả:</h2>
        <p><strong>Xấp xỉ dưới:</strong> {sorted(lower)}</p>
        <p><strong>Xấp xỉ trên:</strong> {sorted(upper)}</p>
    </div>
""")