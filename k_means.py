import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Đọc dữ liệu từ file CSV
df = pd.read_csv('datasets/Mall_Customers.csv')

# Có thể sửa code lại để nhập các cột cần gom cụm ......
# Chọn các cột cần thiết cho K-means
X = df[['Annual_Income_(k$)', 'Spending_Score']].values

# Nhập số cụm từ người dùng
num_clusters = int(input("Nhập số cụm: "))

# Áp dụng thuật toán K-means với số cụm đã nhập
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Vẽ biểu đồ kết quả
plt.figure(figsize=(15, 7))

# Tạo danh sách các màu sắc cho các cụm
colors = ['yellow', 'blue', 'green', 'grey', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta']

# Vẽ các cụm
for i in range(num_clusters):
    sns.scatterplot(x=X[y_kmeans == i, 0], y=X[y_kmeans == i, 1], color=colors[i % len(colors)], label=f'Cluster {i+1}', s=50)

# Vẽ các tâm cụm
sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], color='red', 
                label='Centroids', s=300, marker=',')

plt.grid(False)
plt.title(f'Clusters of customers with {num_clusters} clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()