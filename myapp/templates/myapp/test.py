import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv('myapp/datasets/Groceries_dataset.csv')

basket = (df.groupby(['Member_number','itemDescription'])['Date'].count().unstack().reset_index().fillna(0).set_index('Member_number'))

def encode_units(x):
    if x < 1:
        return 0
    if x >= 1:
        return 1


basket = basket.map(encode_units)

def frequently_bought_together(items):
    # Ensure items is a list
    if isinstance(items, str):
        items = [items]
    
    item_df = basket
    for item in items:
        item_df = item_df.loc[item_df[item] == 1]
    
    # Applying apriori algorithm on item df
    frequent_itemsets = apriori(item_df, min_support=0.4, use_colnames=True)
    
    # Storing association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4, num_itemsets=5)
    
    # Sorting on confidence and support
    rules = rules.sort_values(['confidence', 'support'], ascending=False).reset_index(drop=True)
    
    print('Items frequently bought together with {0}'.format(', '.join(items)))
    
    # Returning top 6 items with highest confidence and support
    return rules['consequents'].unique()[:6]

print(frequently_bought_together('salty snack'))
print(frequently_bought_together(['whole milk', 'yogurt']))