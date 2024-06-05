import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("GroceryStoreDataSet.csv", names=['transaction'], sep=',')

transactions = df['transaction'].apply(lambda x: x.split(",")).tolist()

encoder = TransactionEncoder()
encoded_df = encoder.fit_transform(transactions).toarray()
df_encoded = pd.DataFrame(encoded_df, columns=encoder.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)

frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules)
