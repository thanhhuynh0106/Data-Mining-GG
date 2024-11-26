import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules

dataset = pd.read_csv('datasets/Market_Basket_Optimisation.csv', header=None)
dataset.fillna(0,inplace=True)

transactions = []
for i in range(0,len(dataset)):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20) if str(dataset.values[i,j])!='0'])  

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = fpgrowth(df, min_support=0.025, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3, num_itemsets=5)

def predict_items(purchased_items):
    predicted_items = []
    for index, row in rules.iterrows():
        antecedent = row["antecedents"]
        consequent = row["consequents"]

        if antecedent.issubset(purchased_items):
            predicted_items.append(consequent)

    return predicted_items

purchased_items = {"bacon", "milk", "babies foods"}
predicted_items = predict_items(purchased_items)


print(predicted_items)
