import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import LabelEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import networkx as nx

# Apriori
def apriori_algorithm():
    dataset = pd.read_csv('datasets/Market_Basket_Optimisation.csv', header=None)
    dataset.fillna(0, inplace=True)

    transactions = []
    for i in range(0, len(dataset)):
        transactions.append([str(dataset.values[i, j]) for j in range(0, 20) if str(dataset.values[i, j]) != '0'])

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df, min_support=0.025, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3, num_itemsets=5)
    
    return dataset, df, rules

# Decision Tree
def decision_tree_algorithm():
    dataset = pd.read_csv('datasets/drug200.csv')

    label_encoders = {}
    for column in ['Sex', 'BP', 'Cholesterol', 'Drug']:
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        label_encoders[column] = le

    X = dataset.drop('Drug', axis=1)
    y = dataset['Drug']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return pd.read_csv('datasets/drug200.csv'), model, accuracy, label_encoders

# Giao diện Tkinter
def open_apriori():
    new_window = tk.Toplevel(root)
    new_window.title("Apriori Algorithm")
    create_algorithm_interface(new_window, "apriori")

def open_decision_tree():
    new_window = tk.Toplevel(root)
    new_window.title("Decision Tree Algorithm")
    create_algorithm_interface(new_window, "decision_tree")

def create_algorithm_interface(window, algorithm):
    if algorithm == "apriori":
        raw_data, df, rules = apriori_algorithm()
        
        view_data_button = tk.Button(window, text="View Data", command=lambda: view_data('datasets/Market_Basket_Optimisation.csv'))
        view_data_button.pack(pady=10)
        
        purchased_items_label = tk.Label(window, text="Enter purchased items (comma separated):")
        purchased_items_label.pack(pady=10)
        
        purchased_items_entry = tk.Entry(window, width=50)
        purchased_items_entry.pack(pady=10)
        
        view_results_button = tk.Button(window, text="View Results", command=lambda: view_apriori_results(rules, purchased_items_entry.get()))
        view_results_button.pack(pady=10)
    elif algorithm == "decision_tree":
        raw_data, model, accuracy, label_encoders = decision_tree_algorithm()
        
        view_data_button = tk.Button(window, text="View Data", command=lambda: view_data('datasets/drug200.csv'))
        view_data_button.pack(pady=10)
        
        view_tree_button = tk.Button(window, text="View Tree", command=lambda: view_tree(model, label_encoders))
        view_tree_button.pack(pady=10)
        
        view_results_button = tk.Button(window, text="View Results", command=lambda: view_tree_results(model, accuracy))
        view_results_button.pack(pady=10)

def view_data(file_path):
    data = pd.read_csv(file_path)
    new_window = tk.Toplevel(root)
    new_window.title("View Data")
    
    tree = ttk.Treeview(new_window)
    tree.pack(expand=True, fill='both')
    
    tree["columns"] = list(data.columns)
    tree["show"] = "headings"
    
    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)
    
    for index, row in data.iterrows():
        tree.insert("", "end", values=list(row))

def view_apriori_results(rules, purchased_items_str):
    purchased_items = set(purchased_items_str.split(","))
    predicted_items = []
    for index, row in rules.iterrows():
        antecedent = row["antecedents"]
        consequent = row["consequents"]

        if antecedent.issubset(purchased_items):
            predicted_items.append(consequent)
    
    result_window = tk.Toplevel(root)
    result_window.title("Predicted Items")
    
    result_label = tk.Label(result_window, text=f"Predicted items: {predicted_items}")
    result_label.pack(pady=10)

def view_tree(model, label_encoders):
    plt.figure(figsize=(20,20))
    plot_tree(model, 
              feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], 
              class_names=label_encoders['Drug'].classes_, 
              filled=True, 
              rounded=True, 
              proportion=False, 
              precision=2, 
              fontsize=10)
    plt.show()

def view_tree_results(model, accuracy):
    result_window = tk.Toplevel(root)
    result_window.title("Decision Tree Results")
    
    result_label = tk.Label(result_window, text=f"Accuracy: {accuracy:.2f}")
    result_label.pack(pady=10)
    
    sample = pd.DataFrame([[30, 1, 2, 1, 15.5],
                           [24, 0, 0, 0, 12.31],
                           [42, 0, 1, 1, 9.3],
                           [60, 1, 0, 0, 7.655]
                          ], columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

    predicted_drug = model.predict(sample)
    sample_result_label = tk.Label(result_window, text=f"Predicted medicine of the new patient: {predicted_drug}")
    sample_result_label.pack(pady=10)

root = tk.Tk()
root.title("Algorithm Selector")

apriori_button = tk.Button(root, text="Apriori Algorithm", command=open_apriori)
apriori_button.pack(pady=10)

decision_tree_button = tk.Button(root, text="Decision Tree Algorithm", command=open_decision_tree)
decision_tree_button.pack(pady=10)

root.mainloop()