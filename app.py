import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit Page Configuration
st.set_page_config(page_title="Shopping Trends Association Rules", layout="wide")

st.title("🛒 Shopping Trends - Association Rule Mining")

# Load Data
df = pd.read_csv('shopping_trends.csv')
df['transactions'] = df['Item Purchased'] + '*' + df['Category']

# Dummy Variables
X = df['transactions'].str.get_dummies(sep='*')

# Item Frequency Analysis
item_counts = X.sum().sort_values(ascending=False)
top_items = item_counts.head(10)

st.subheader("🔝 Top 10 Most Frequent Items")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x=top_items.values, y=top_items.index, palette='Blues_r', ax=ax1)
ax1.set_title("Top 10 Most Frequent Items")
ax1.set_xlabel("Frequency")
st.pyplot(fig1)

# Optional: Heatmap
st.subheader("🔥 Item Co-occurrence Heatmap")
top_10_cols = X[top_items.index]
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(top_10_cols.corr(), annot=True, cmap='coolwarm', ax=ax2)
ax2.set_title("Item Co-occurrence Heatmap (Top 10 Items)")
st.pyplot(fig2)

# User Inputs
st.sidebar.header("🔧 Parameters")
min_support = st.sidebar.number_input("Minimum Support", value=0.01, min_value=0.001, max_value=1.0, step=0.01)
min_lift = st.sidebar.number_input("Minimum Lift", value=1.2, min_value=0.1, max_value=5.0, step=0.1)

# Apriori
frequent_itemsets = apriori(X, min_support=min_support, use_colnames=True, max_len=4)
frequent_itemsets.sort_values('support', ascending=False, inplace=True)

st.subheader("📦 Top 10 Frequent Itemsets")
fig3, ax3 = plt.subplots(figsize=(12, 7))
top_sets = frequent_itemsets.head(10)
labels = [' + '.join(list(i)) for i in top_sets.itemsets]
sns.barplot(x=top_sets.support, y=labels, palette='Reds_r', ax=ax3)
ax3.set_title("Top 10 Frequent Itemsets")
ax3.set_xlabel("Support")
st.pyplot(fig3)

# Association Rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
rules = rules[rules['confidence'] >= 0.5]
rules.sort_values('lift', ascending=False, inplace=True)

st.subheader("📈 Top 10 Association Rules by Lift")
st.dataframe(rules.head(10))

# Confidence and Lift Distribution
st.subheader("📊 Rule Metrics Distribution")
fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(rules['confidence'], bins=10, color='skyblue', ax=ax4)
ax4.set_title("Confidence Distribution")
sns.histplot(rules['lift'], bins=10, color='lightcoral', ax=ax5)
ax5.set_title("Lift Distribution")
st.pyplot(fig4)

# Redundant Rule Removal
def to_list(i): return sorted(list(i))
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(x) for x in set(tuple(x) for x in rules_sets)]
index_rules = [rules_sets.index(x) for x in unique_rules_sets]
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy.sort_values('lift', ascending=False, inplace=True)

st.subheader("🚫 Top 10 Non-Redundant Rules")
st.dataframe(rules_no_redundancy.head(10))

# Save to CSV
rules_no_redundancy.to_csv('filtered_association_rules.csv', index=False)
st.success("✅ Filtered rules saved to `filtered_association_rules.csv`")

