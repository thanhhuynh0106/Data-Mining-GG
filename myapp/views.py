from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from datetime import datetime
import math
from collections import defaultdict



def home_content(request):
    # Hàm hiển thị trang chủ
    return render(request, 'myapp/home.html')
def home(request):
    # Hàm hiển thị trang chủ
    return render(request, 'myapp/home_second.html')  

def upload_view(request):
    # Hàm hiển thị trang upload
    return render(request, 'myapp/upload_data.html')
timestamp1 = datetime.now().strftime('%Y%m%d%H%M%S')
def upload_data_csv(request):
    if request.method == 'POST':
        algorithm = request.POST.get('algorithm')
        csv_file = request.FILES['csv_file']
        file_extension = os.path.splitext(csv_file.name)[1]
        file_path = f"myapp/datasets/{algorithm}_new{timestamp1}{file_extension}"
        df = pd.read_csv(csv_file)
        df.to_csv(file_path, index=False)
        return HttpResponse("<p style='color: green;'>File uploaded successfully!</p>")
    # if request.method == 'POST' and request.FILES.get('csv_file'):
    #     csv_file = request.FILES['csv_file']
    #     file_path = "myapp/datasets/uploaded_data_1.csv"
    #     df = pd.read_csv(csv_file)
    #     df.to_csv(file_path, index=False)
    #     return HttpResponse("<p style='color: green;'>File uploaded successfully!</p>")
    # return HttpResponse("<p style='color: red;'>No file uploaded.</p>")


def view_data(request):
    # Hàm xem dữ liệu đã upload
    data_json = request.session.get('data')
    if data_json:
        data = pd.read_json(data_json)
        html = data.to_html()
        return HttpResponse(html)
    return HttpResponse('No data available')

def view_results(request):
    # Hàm hiển thị kết quả tính toán
    results = "Kết quả tính toán"
    return HttpResponse(results)

def show_data(request):
    # Hàm hiển thị dữ liệu từ file CSV
    try:
        df_show = pd.read_csv(file_path1)
        data_html = df_show.to_html(index=False, classes="table table-bordered", border=0, justify='center')
        data_html = data_html.replace('<th>', '<th style="text-align: center;">')
        data_html = data_html.replace('<td>', '<td style="text-align: center;">')
        return HttpResponse(data_html)
    except Exception as e:
        return HttpResponse(f"<p style='color: red;'>Error loading data: {e}</p>")

##### TẬP THÔ
## TÍNH XẤP XỈ
def approximation_view(request):
    # Hàm hiển thị trang approximation
    return render(request, 'myapp/approximation.html')

file_path1 = 'myapp/datasets/reduct.csv'  # Đảm bảo file này nằm cùng thư mục hoặc chỉ định đúng đường dẫn

def calculate_approximation(request):
    # Hàm tính toán xấp xỉ
    data = pd.read_csv(file_path1)

    def indiscernibility_relation(data, attributes):
        # Hàm tính quan hệ bất khả phân biệt
        equivalence_classes = {}
        for idx, row in data.iterrows():
            key = tuple(row[attr] for attr in attributes)
            if key not in equivalence_classes:
                equivalence_classes[key] = []
            equivalence_classes[key].append(idx)
        return equivalence_classes

    eq_classes = indiscernibility_relation(data, ['Age', 'Sessions'])

    def lower_approximation(eq_classes, target_set):
        # Hàm tính xấp xỉ dưới
        lower = []
        for eq_class in eq_classes.values():
            if set(eq_class).issubset(target_set):
                lower.extend(eq_class)
        return set(lower)

    def upper_approximation(eq_classes, target_set):
        # Hàm tính xấp xỉ trên
        upper = []
        for eq_class in eq_classes.values():
            if set(eq_class).intersection(target_set):
                upper.extend(eq_class)
        return set(upper)

    target_set = set(data[data['Pass'] == 'Yes'].index)
    lower = lower_approximation(eq_classes, target_set)
    upper = upper_approximation(eq_classes, target_set)
    accuracy = len(lower) / len(upper) if upper else 0

    lower_yes = lower_approximation(eq_classes, set(data[data['Pass'] == 'Yes'].index))
    lower_no = lower_approximation(eq_classes, set(data[data['Pass'] == 'No'].index))
    count_lower_yes = len(lower_yes)
    count_lower_no = len(lower_no)
    total_count = len(data)
    dependency_ratio = (count_lower_yes + count_lower_no) / total_count
    dependency_ratio = round(dependency_ratio, 2)

    return HttpResponse(f"""
        <h2>Kết quả:</h2>
        <p><strong>Quan hệ bất khả phân biệt giữa Age và Sessions:</strong><br> {eq_classes}</p>
        <p><strong>Xấp xỉ dưới:</strong> {sorted(lower)}</p>
        <p><strong>Xấp xỉ trên:</strong> {sorted(upper)}</p>
        <p><strong>Độ chính xác:</strong> {accuracy}</p>
        <p><strong>Khảo sát sự phụ thuộc của Pass vào Age và Sessions:</strong> {dependency_ratio}</p>
    </div>
""")

## Reduct
def show_data_reduct(request):
    # Hàm hiển thị dữ liệu reduct
    df = pd.read_excel('myapp/datasets/reduct_r.xlsx')
    df_show = df.head(20)
    data_html = df_show.to_html(index=False, classes="table table-bordered", border=1, justify='center')
    data_html = data_html.replace('<th>', '<th style="text-align: center;">')
    data_html = data_html.replace('<td>', '<td style="text-align: center;">')
    return HttpResponse(data_html)

def reduct_view(request):
    # Hàm hiển thị trang reduct
    return render(request, 'myapp/reduct.html')

def calculate_reduct(request):
    # Hàm tính toán reduct
    file_path = 'myapp/datasets/reduct_r.xlsx'
    data = pd.read_excel(file_path)

    df_encoded = pd.get_dummies(data.drop('Kết quả', axis=1), drop_first=True)
    y = data['Kết quả'].apply(lambda x: 1 if x == 'Bị rám' else 0)
    model = DecisionTreeClassifier()
    rfe = RFE(model, n_features_to_select=2)
    fit = rfe.fit(df_encoded, y)
    selected_features = df_encoded.columns[fit.support_]
    num_features_list = [2, 3]
    results = []

    for num_features in num_features_list:
        rfe = RFE(model, n_features_to_select=num_features)
        fit = rfe.fit(df_encoded, y)
        scores = cross_val_score(model, df_encoded[df_encoded.columns[fit.support_]], y, cv=5)
        results.append({'num_features': num_features,
                        'selected_features': df_encoded.columns[fit.support_].tolist(),
                        'mean_score': scores.mean()})

    res_list = []
    for result in results:
        res_list.append(result['selected_features'])

    return HttpResponse(f"""
        <h2>Kết quả:</h2>
        <p><strong>Các thuộc tính được chọn:</strong> {res_list}</p>
    </div>
""")


##### APRIORI
import pandas as pd
from itertools import combinations

def apriori(df, min_support=0.1):
    from collections import defaultdict

    itemset_support = defaultdict(int)
    customers = df['Member_number'].unique()

    for customer in customers:
        items = df[df['Member_number'] == customer]['itemDescription'].unique()
        for r in range(1, len(items) + 1):
            for itemset in combinations(items, r):
                itemset_support[tuple(sorted(itemset))] += 1

    num_customers = len(customers)
    frequent_itemsets = {itemset: support / num_customers for itemset, support in itemset_support.items() if support / num_customers >= min_support}

    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset, itemset_support in frequent_itemsets.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = tuple(sorted(antecedent))
                    consequent = tuple(sorted(set(itemset) - set(antecedent)))
                    antecedent_support = frequent_itemsets.get(antecedent)
                    if antecedent_support:
                        confidence = itemset_support / antecedent_support
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, confidence))
    return rules

def calculate_apriori(request):
    import warnings
    warnings.filterwarnings("ignore")

    if request.method == 'POST':
        min_support = float(request.POST.get('min_support', 0.05))
        min_confidence = float(request.POST.get('min_confidence', 0.4))

    df = pd.read_csv('myapp/datasets/Groceries_dataset.csv')

    frequent_itemsets = apriori(df, min_support=min_support)

    if not frequent_itemsets:
        return HttpResponse("Không có tập phổ biến nào được tìm thấy với min_support này.")

    rules = generate_rules(frequent_itemsets, min_confidence=min_confidence)

    frequent_itemsets_html = "<h2>Frequent Itemsets:</h2><ul>"
    frequent_itemsets_html += "".join(f"<li>{itemset}: {support:.2f}</li>" for itemset, support in frequent_itemsets.items())
    frequent_itemsets_html += "</ul>"

    rules_html = "<h2>Association Rules:</h2><ul>"
    rules_html += "".join(f"<li>{antecedent} -> {consequent} (confidence: {confidence:.2f})</li>" for antecedent, consequent, confidence in rules)
    rules_html += "</ul>"

    return HttpResponse(frequent_itemsets_html + rules_html)


def apriori_view(request):
    # Hàm hiển thị trang apriori
    return render(request, 'myapp/apriori.html')

def show_data_apriori(request):
    # Hàm hiển thị dữ liệu apriori
    df = pd.read_csv('myapp/datasets/Groceries_dataset.csv')
    df_show = df.head(20)
    data_html = df_show.to_html(index=False, classes="table table-bordered", border=0, justify='center')
    data_html = data_html.replace('<th>', '<th style="text-align: center;">')
    data_html = data_html.replace('<td>', '<td style="text-align: center;">')
    return HttpResponse(data_html)

##### Phân lớp
## Cây quyết định

def decision_tree_view(request):
    # Hàm hiển thị trang decision tree
    return render(request, 'myapp/decisiontree.html')

def show_data_decision_tree(request):
    # Hàm hiển thị dữ liệu decision tree
    try:
        df = pd.read_csv(f'myapp/datasets/decisiontree_new{timestamp1}.csv')
    except FileNotFoundError:
        df = pd.read_csv('myapp/datasets/drug200.csv')
    df_show = df.head(20)
    data_html = df_show.to_html(index=False, classes="table table-bordered", border=0, justify='center')
    data_html = data_html.replace('<th>', '<th style="text-align: center;">')
    data_html = data_html.replace('<td>', '<td style="text-align: center;">')
    return HttpResponse(data_html)


def entropy(freq):
    # Hàm tính entropy
    prob = freq / float(freq.sum())
    return -np.sum(prob * np.log2(prob + 1e-9))

def calculate_entropy(target, ids):
    # Hàm tính entropy của 1 tập hợp
    if len(ids) == 0:
        return 0
    freq = np.array(target.iloc[ids].value_counts())
    return entropy(freq)

def set_label(target, ids):
    # Hàm thiết lập nhãn
    return target.iloc[ids].mode()[0]

def split_node(data, target, ids, min_samples_split, min_gain):
    # Hàm chia nút
    best_gain = 0
    best_splits = []
    best_attribute = None
    order = None
    sub_data = data.iloc[ids, :]
    for i, att in enumerate(data.columns):
        values = data.iloc[ids, i].unique().tolist()
        if len(values) == 1:
            continue
        splits = []
        for val in values:
            sub_ids = sub_data.index[sub_data[att] == val].tolist()
            splits.append(sub_ids)
        if min(map(len, splits)) < min_samples_split:
            continue
        HxS = 0
        for split in splits:
            HxS += len(split) * calculate_entropy(target, split) / len(ids)
        gain = calculate_entropy(target, ids) - HxS
        if gain < min_gain:
            continue
        if gain > best_gain:
            best_gain = gain
            best_splits = splits
            best_attribute = att
            order = values
    return best_attribute, order, best_splits

def build_tree(data, target, ids, depth, max_depth, min_samples_split, min_gain):
    # Hàm xây dựng cây
    entropy = calculate_entropy(target, ids)
    if depth >= max_depth or entropy <= min_gain:
        return {'label': set_label(target, ids)}
    split_attribute, order, splits = split_node(data, target, ids, min_samples_split, min_gain)
    if not splits:
        return {'label': set_label(target, ids)}
    children = []
    for split in splits:
        child = build_tree(data, target, split, depth + 1, max_depth, min_samples_split, min_gain)
        children.append(child)
    return {'split_attribute': split_attribute, 'order': order, 'children': children}

def predict(tree, new_data):
    # Hàm dự đoán
    labels = []
    for _, x in new_data.iterrows():
        node = tree
        while 'children' in node:
            idx = node['order'].index(x[node['split_attribute']])
            node = node['children'][idx]
        labels.append(node['label'])
    return labels

def plot_tree(node, depth=0, pos=(0, 0), dx=1.5, dy=1.5, ax=None):
    # Hàm vẽ cây
    if ax is None:
        ax = plt.gca()
    
    x, y = pos
    node_text = f"{node['split_attribute']}" if 'label' not in node else f"{node['label']}"
    
    if 'label' in node:
        color = "lightgrey"
        text_color = "red" if node['label'] == "No" else "black"
    else:
        color = "limegreen"
        text_color = "white"

    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=color)
    ax.text(x, y, node_text, ha='center', va='center', color=text_color, bbox=bbox_props, fontsize=12, fontweight='bold')

    if 'children' in node:
        num_children = len(node['children'])
        for i, child in enumerate(node['children']):
            child_x = x + (i - (num_children - 1) / 2) * dx / (2 ** depth)
            child_y = y - dy  

            ax.plot([x, child_x], [y - 0.1, child_y + 0.1], 'k-', lw=1.5)

            edge_label = node['order'][i]
            ax.text((x + child_x) / 2, (y + child_y) / 2, edge_label,
                    ha='center', va='center', fontsize=10, color="black",
                    bbox=dict(boxstyle="round,pad=0.2", edgecolor="none", facecolor="lightgrey"))
            
            plot_tree(child, depth + 1, (child_x, child_y), dx, dy, ax)

def calculate_decision_tree(request):
    # Hàm tính toán cây quyết định
    try:
        df = pd.read_csv(f'myapp/datasets/decisiontree_new{timestamp1}.csv')
    except FileNotFoundError:
        df = pd.read_csv('myapp/datasets/drug200.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    tree = build_tree(X, y, list(range(len(X))), 0, max_depth=3, min_samples_split=2, min_gain=1e-4)

    predicted_labels = predict(tree, X)

    # Tính toán độ chính xác
    accuracy = sum(1 for actual, predicted in zip(y, predicted_labels) if actual == predicted) / len(y) * 100

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_title("Decision Tree Visualization", fontsize=16)
    plot_tree(tree, dx=3.0, dy=1.8, ax=ax)
    ax.axis('off')

    # Tạo tên tệp hình ảnh duy nhất bằng cách sử dụng thời gian hiện tại
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    image_path = f'myapp/static/pictures/decision_tree_{timestamp}.png'
    plt.savefig(image_path)
    plt.close()

    return HttpResponse(f"""
        <h2>Kết quả:</h2>
        <img src="/static/pictures/decision_tree_{timestamp}.png" alt="Decision Tree">
    """)


## Navie Bayes
def naivebayes_view(request):
    # Hàm hiển thị trang naive bayes
    return render(request, 'myapp/naivebayes.html')

def show_data_naivebayes(request):
    # Hàm hiển thị dữ liệu naive bayes
    df = pd.read_csv('myapp/datasets/drug200.csv')
    df_show = df.head(20)
    data_html = df_show.to_html(index=False, classes="table table-bordered", border=0, justify='center')
    data_html = data_html.replace('<th>', '<th style="text-align: center;">')
    data_html = data_html.replace('<td>', '<td style="text-align: center;">')
    return HttpResponse(data_html)

def read_csv(file_path):
    # Hàm đọc file CSV
    df = pd.read_csv(file_path)
    return df.values.tolist()

def split_features_labels(data):
    # Hàm chia features và labels
    features = [row[:-1] for row in data]
    labels = [row[-1] for row in data]
    return features, labels

def calculate_prior_probs(labels):
    # Hàm tính xác suất tiên nghiệm
    total_samples = len(labels)
    class_counts = defaultdict(int)
    for label in labels:
        class_counts[label] += 1
    prior_probs = {label: count / total_samples for label, count in class_counts.items()}
    return prior_probs, set(labels)

def calculate_conditional_probs(features, labels, classes, laplace_smoothing=False):
    # Hàm tính xác suất có điều kiện
    conditional_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    feature_value_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    label_counts = defaultdict(int)

    for feature_index in range(len(features[0])):
        for row_index in range(len(features)):
            label = labels[row_index]
            feature_value = features[row_index][feature_index]
            feature_value_counts[feature_index][label][feature_value] += 1
            label_counts[label] += 1

    for feature_index in range(len(features[0])):
        for label in classes:
            total_label_features = label_counts[label]
            unique_feature_values = len(set(row[feature_index] for row in features))
            for feature_value, count in feature_value_counts[feature_index][label].items():
                if laplace_smoothing:
                    conditional_probs[feature_index][label][feature_value] = (count + 1) / (total_label_features + unique_feature_values)
                else:
                    conditional_probs[feature_index][label][feature_value] = count / total_label_features

            if laplace_smoothing:
                for feature_value in set(row[feature_index] for row in features):
                    if feature_value not in feature_value_counts[feature_index][label]:
                        conditional_probs[feature_index][label][feature_value] = 1 / (total_label_features + unique_feature_values)

    return conditional_probs

def fit_naive_bayes(features, labels, laplace_smoothing=False):
    # Hàm fit naive bayes
    prior_probs, classes = calculate_prior_probs(labels)
    conditional_probs = calculate_conditional_probs(features, labels, classes, laplace_smoothing)
    return prior_probs, conditional_probs, classes

def predict_naive_bayes(features, prior_probs, conditional_probs, classes):
    # Hàm dự đoán naive bayes
    predictions = []
    for x in features:
        class_probs = {}
        for label in classes:
            class_probs[label] = math.log(prior_probs[label])
            for feature_index, feature_value in enumerate(x):
                if feature_value in conditional_probs[feature_index][label]:
                    class_probs[label] += math.log(conditional_probs[feature_index][label][feature_value])
                else:
                    class_probs[label] += math.log(1 / (sum(conditional_probs[feature_index][label].values()) + len(conditional_probs[feature_index][label])))
        best_label = max(class_probs, key=class_probs.get)
        predictions.append(best_label)
    return predictions

def calculate_accuracy(predictions, labels):
    # Hàm tính độ chính xác
    correct = 0
    for pred, label in zip(predictions, labels):
        if pred == label:
            correct += 1
    accuracy = (correct / len(labels)) * 100
    return accuracy

def calculate_naivebayes(request):
    # Hàm tính toán naive bayes
    data = pd.read_csv('myapp/datasets/drug200.csv').values.tolist()

    if request.method == 'POST':
        # Lấy dữ liệu từ form input
        age = int(request.POST.get('age'))
        sex = request.POST.get('sex')
        BP = request.POST.get('BP')
        Cholesterol = request.POST.get('Cholesterol')
        Na_to_K = float(request.POST.get('Na_to_K'))

        # Tạo dữ liệu kiểm tra
        test_data = [[age, sex, BP, Cholesterol, Na_to_K]]
    features, labels = split_features_labels(data)
    
    # Without Laplace smoothing
    prior_probs, conditional_probs, classes = fit_naive_bayes(features, labels, laplace_smoothing=False)
    predictions_no_smoothing = predict_naive_bayes(test_data, prior_probs, conditional_probs, classes)
    original_predictions_no_smoothing = predict_naive_bayes(features, prior_probs, conditional_probs, classes)
    accuracy_no_smoothing = calculate_accuracy(original_predictions_no_smoothing, labels)

    return HttpResponse(f"""
        <h2>Kết quả:</h2>
        <h3>Không làm trơn Laplace:</h3>
        <p><strong>Dự đoán:</strong> {predictions_no_smoothing}</p>
    """)

def calculate_naivebayes_smoothing(request):
    # Hàm tính toán naive bayes có làm trơn Laplace
    data = pd.read_csv('myapp/datasets/drug200.csv').values.tolist()

    if request.method == 'POST':
        # Lấy dữ liệu từ form input
        age = int(request.POST.get('age'))
        sex = request.POST.get('sex')
        BP = request.POST.get('BP')
        Cholesterol = request.POST.get('Cholesterol')
        Na_to_K = float(request.POST.get('Na_to_K'))

        # Tạo dữ liệu kiểm tra
        test_data = [[age, sex, BP, Cholesterol, Na_to_K]]

    features, labels = split_features_labels(data)
    
    # With Laplace smoothing
    prior_probs_ls, conditional_probs_ls, classes_ls = fit_naive_bayes(features, labels, laplace_smoothing=True)
    predictions_smoothing = predict_naive_bayes(test_data, prior_probs_ls, conditional_probs_ls, classes_ls)
    original_predictions_smoothing = predict_naive_bayes(features, prior_probs_ls, conditional_probs_ls, classes_ls)
    accuracy_smoothing = calculate_accuracy(original_predictions_smoothing, labels)

    return HttpResponse(f"""
        <h2>Kết quả:</h2>
        <h3>Làm trơn Laplace:</h3>
        <p><strong>Dự đoán:</strong> {predictions_smoothing}</p>
    """)


##### Gom cụm
## K-means

def kmeans_view(request):
    # Hàm hiển thị trang k-means
    return render(request, 'myapp/kmeans.html')

def show_data_kmeans(request):
    # Hàm hiển thị dữ liệu k-means
    try:
        df = pd.read_csv(f'myapp/datasets/kmeans_new{timestamp1}.csv')
    except FileNotFoundError:
        df = pd.read_csv('myapp/datasets/Mall_Customers.csv')
    df_show = df.head(20)
    data_html = df_show.to_html(index=False, classes="table table-bordered", border=0, justify='center')
    data_html = data_html.replace('<th>', '<th style="text-align: center;">')
    data_html = data_html.replace('<td>', '<td style="text-align: center;">')
    return HttpResponse(data_html)


def initialize_centroids(X, k):
    # Hàm khởi tạo tâm
    np.random.seed(42)
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_clusters(X, centroids):
    # Hàm gán cụm
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X, labels, k):
    # Hàm cập nhật tâm
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

def kmeans(X, k, max_iters=100):
    # Hàm k-means
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def calculate_kmeans(request):
    # Hàm tính toán k-means
    if request.method == 'POST':
        num_clusters = int(request.POST.get('num_clusters', 5))

    try:
        data = pd.read_csv(f'myapp/datasets/kmeans_new{timestamp1}.csv')
    except FileNotFoundError:
        data = pd.read_csv('myapp/datasets/Mall_Customers.csv')

    if 'Annual_Income_(k$)' in data.columns and 'Spending_Score' in data.columns:
        features = data[['Annual_Income_(k$)', 'Spending_Score']].values
    else:
        features = data.values

    centroids, labels = kmeans(features, num_clusters)
    data['Cluster'] = labels

    # Tạo tên tệp hình ảnh duy nhất bằng cách sử dụng thời gian hiện tại
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    image_path = f'myapp/static/pictures/kmeans_clusters_{timestamp}.png'
    
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', marker='x', label='Centroids')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('K-means Clustering')
    plt.legend()
    plt.savefig(image_path)
    plt.close()

    return HttpResponse(f"""
        <h2>Kết quả:</h2>
        <p><strong>Tâm của các cụm:</strong> {centroids}</p>
        <img src="/static/pictures/kmeans_clusters_{timestamp}.png" alt="K-means Clustering">
    """)


## Kohonen
def kohonen_view(request):
    # Hàm hiển thị trang kohonen
    return render(request, 'myapp/kohonen.html')

def show_data_kohonen(request):
    # Hàm hiển thị dữ liệu kohonen
    try:
        df = pd.read_csv(f'myapp/datasets/kohonen_new{timestamp1}.csv')
    except FileNotFoundError:
        df = pd.read_csv('myapp/datasets/Mall_Customers.csv')
    df_show = df.head(20)
    data_html = df_show.to_html(index=False, classes="table table-bordered", border=0, justify='center')
    data_html = data_html.replace('<th>', '<th style="text-align: center;">')
    data_html = data_html.replace('<td>', '<td style="text-align: center;">')
    return HttpResponse(data_html)


def calculate_kohonen(request):
    # Hàm tính toán kohonen
    if request.method == 'POST':
        m = int(request.POST.get('num_rows', 10))
        n = int(request.POST.get('num_cols', 10))
        learning_rate = float(request.POST.get('learning_rate', 0.1))
        num_iters = int(request.POST.get('num_iters', 100))
        radius = float(request.POST.get('radius', 1.0))

    file_path = f'myapp/datasets/kohonen_new{timestamp1}.csv'
    if not os.path.exists(file_path):
        file_path = 'myapp/datasets/Mall_Customers.csv'
        
    data = pd.read_csv(file_path, encoding='utf-8')
        
    if 'Annual_Income_(k$)' in data.columns and 'Spending_Score' in data.columns:
        features = data[['Annual_Income_(k$)', 'Spending_Score']].values
    else:
        features = data.values

    def initialize_som(m, n, input_dim):
        # Hàm khởi tạo SOM
        np.random.seed(42)
        return np.random.rand(m, n, input_dim)

    def find_bmu(som, x):
        # Hàm tìm BMU
        distances = np.linalg.norm(som - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def update_weights(som, x, bmu_idx, learning_rate, radius):
        # Hàm cập nhật trọng số
        for i in range(som.shape[0]):
            for j in range(som.shape[1]):
                if np.linalg.norm(np.array([i, j]) - bmu_idx) <= radius:
                    som[i, j] += learning_rate * (x - som[i, j])

    def train_som(X, som, num_iters, learning_rate, radius):
        # Hàm huấn luyện SOM
        for iter in range(num_iters):
            for x in X:
                bmu_idx = find_bmu(som, x)
                update_weights(som, x, bmu_idx, learning_rate, radius)
            learning_rate *= 0.9
            radius *= 0.9
        return som

    som = initialize_som(m, n, features.shape[1])
    som = train_som(features, som, num_iters, learning_rate, radius)

    labels = np.array([find_bmu(som, x) for x in features])
    data['Cluster'] = [str(label) for label in labels]

    unique_labels = np.unique(labels, axis=0)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = np.all(labels == label, axis=1)
        plt.scatter(features[mask, 0], features[mask, 1], color=colors(i), label=f'Cluster {i}')

    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score')
    plt.title('Kohonen SOM Clustering')
    plt.legend()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    image_path = f'myapp/static/pictures/kohonen_clusters_{timestamp}.png'
    plt.savefig(image_path)
    plt.close()

    return HttpResponse(f"""
        <h2>Kết quả:</h2>
        <img src="/static/pictures/kohonen_clusters_{timestamp}.png" alt="Kohonen SOM Clustering">
    """)


def delete_pictures():
    # Hàm xóa các hình ảnh đã in ra
    pictures_path = 'myapp/static/pictures/*.png'
    pictures = glob.glob(pictures_path)
    for picture in pictures:
        os.remove(picture)
    return
delete_pictures()

## xóa data vừa mới thêm vào
def delete_uploaded_data():
      try:
        for file in glob.glob('myapp/datasets/*_new*.csv'):
             os.remove(file)
      except FileNotFoundError:

           pass
delete_uploaded_data()