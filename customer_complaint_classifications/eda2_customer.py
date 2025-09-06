
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import chi2_contingency


# # plotting aesthetics
# sns.set(style="whitegrid")
# RND = 42

# # load (update path)
# customer_data = pd.read_csv(r'/Users/sagnikgupta/Desktop/Python_Project/NLP Projects/rows.csv', low_memory=False)
# print("Rows, cols:", customer_data.shape)
# print("Columns:", customer_data.columns.tolist())


# # 1. Prepare
# customer_data['Date received'] = pd.to_datetime(customer_data['Date received'], errors='coerce')

# # basic missingness
# missing = customer_data.isnull().sum().sort_values(ascending=False)
# pct_missing = (missing / len(customer_data) * 100).round(2)

# # top lists
# top_products = customer_data['Product'].value_counts().head(10)
# top_issues = customer_data['Issue'].value_counts().head(10)
# top_companies = customer_data['Company'].value_counts().head(10)
# top_states = customer_data['State'].value_counts().head(10)
# complaints_over_time = customer_data.groupby(customer_data['Date received'].dt.to_period("M")).size()


# a = customer_data.groupby(['Company'])['Product'].count().sort_values(ascending=False).head(10).reset_index()
# a["Pct"]= a['Product']/a["Product"].sum()*100
# a

# # Plot pair 1: missing bars + target distribution
# fig, axes = plt.subplots(1,2, figsize=(14,5))
# pct_missing[pct_missing>0].head(10).plot(kind='barh', ax=axes[0], color='tomato')
# axes[0].set_title("Top 10 Columns: % Missing")
# axes[0].invert_yaxis()

# top_products.plot(kind='bar', ax=axes[1], color='steelblue')
# axes[1].set_title("Top Products (by complaint count)")
# axes[1].set_ylabel("Count")
# plt.suptitle("Step 1: Data Quality & Target Overview", fontsize=14)
# plt.tight_layout()
# plt.show()

# # Plot pair 2: Issues and Companies
# fig, axes = plt.subplots(1,2, figsize=(15,12))
# top_issues.plot(kind='barh', ax=axes[0], color='purple')
# axes[0].set_title("Top Issues (overall)")

# top_companies.plot(kind='bar', ax=axes[1], color='darkorange')
# axes[1].set_title("Top Companies by Volume")
# axes[1].set_ylabel("Count")
# plt.suptitle("Step 2: Nature of Complaints", fontsize=14)
# plt.tight_layout()
# plt.show()

# # Plot pair 3: States and Trend
# fig, axes = plt.subplots(1,2, figsize=(14,5))
# top_states.plot(kind='bar', ax=axes[0], color='green')
# axes[0].set_title("Top States by Complaints")

# complaints_over_time.plot(ax=axes[1], color='red')
# axes[1].set_title("Complaints Over Time (monthly)")
# plt.suptitle("Step 3: Geography & Time", fontsize=14)
# plt.tight_layout()
# plt.show()

# # pie: disputes proportion if column exists
# if 'Consumer disputed?' in customer_data.columns:
#     plt.figure(figsize=(5,5))
#     customer_data['Consumer disputed?'].fillna('Unknown').value_counts().plot(
#         kind='pie', autopct='%1.1f%%', startangle=90)
#     plt.title("Consumer disputed? (proportions)")
#     plt.ylabel("")
#     plt.show()




# '''
# Narration:
# “Show missingness — note Consumer complaint narrative high missingness (~70%). 
# For segmentation checks we will still use rows with Product present.
# Show product & issue distributions to orient stakeholders (which product(s) dominate).
# Show top companies and states to reveal concentration and hotspots; time series to catch spikes (e.g., COVID).”

# '''

# # 2. Prepare segmentation dataset (keep Product & segments)
# seg_cols = ['Product','Issue','Company','State','Date received']
# seg_df = customer_data[seg_cols].copy()

# # drop rows with no Product (target is required)
# seg_df = seg_df.dropna(subset=['Product']).copy()

# # fill NA for segment columns with 'Unknown' so categories included
# seg_df['Company'] = seg_df['Company'].fillna('Unknown')
# seg_df['State'] = seg_df['State'].fillna('Unknown')
# seg_df['Year'] = pd.to_datetime(seg_df['Date received'], errors='coerce').dt.year.fillna(-1).astype(int)

# print("Segmentation df shape:", seg_df.shape)
# print("Unique products:", seg_df['Product'].nunique())

# '''
# company==HSBC--PRODUCTS

# '''
# # ==========================================================================================
# # 3. Statistical tests with interpretation
# # ==========================================================================================

# def cramers_v_with_p(tab):
#     chi2, p, dof, expected = chi2_contingency(tab)
#     n = tab.values.sum()
#     r, k = tab.shape
#     v = np.sqrt((chi2/n) / (min(r-1, k-1)))
#     return v, p, chi2, dof

# candidates = {
#     'Company': seg_df['Company'],
#     'State': seg_df['State'],
#     'Year': seg_df['Year']
# }

# results = {}
# for name, col in candidates.items():
#     # Handle large categories
#     if name == 'Company':
#         topN = 30
#         keep = seg_df['Company'].value_counts().head(topN).index
#         sub = seg_df[seg_df['Company'].isin(keep)]
#         tab = pd.crosstab(sub['Company'], sub['Product']) #--contingency table 
#     elif name == 'State':
#         topN = 20
#         keep = seg_df['State'].value_counts().head(topN).index
#         sub = seg_df[seg_df['State'].isin(keep)]
#         tab = pd.crosstab(sub['State'], sub['Product'])
#     else:  # Year
#         tab = pd.crosstab(seg_df['Year'], seg_df['Product'])

#     # Run test
#     v, p, chi2, dof = cramers_v_with_p(tab)
#     results[name] = {
#         'cramers_v': v,
#         'p_value': p,
#         'chi2': chi2,
#         'dof': dof,
#         'rows_in_tab': tab.values.sum()
#     }

#     # Print results
#     print(f"\n{name} vs Product → Cramer's V = {v:.3f} | p = {p:.3e} | rows = {tab.values.sum()}")

#     # Interpretation with If–Else
#     if p < 0.05 and v >= 0.3:
#         print(f"Conclusion → Strong evidence: segmentation by {name.upper()} is justified.")
#     elif p < 0.05 and v >= 0.1:
#         print(f"Conclusion → Some evidence: segmentation by {name.upper()} may help.")
#     else:
#         print(f"Conclusion → Weak/No evidence: segmentation by {name.upper()} not needed.")

# '''
# EDA shows a strong association between Company and Product. 
# This suggests that Products are unevenly distributed across Companies. 
# However, since our modeling objective is to classify Products based on the narrative text, 
# we will not segment by Company to avoid information leakage. Instead, we will build a global model on narratives.
# '''

# results['Company'], results['State'], results['Year']


# pd.DataFrame(results).T

# # chi-- p value, test statistic, 












import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Plotting aesthetics
sns.set(style="whitegrid")
RND = 42

# Load data (update path accordingly)
data_path = r'/Users/sagnikgupta/Desktop/Python_Project/NLP Projects/rows.csv'
customer_data = pd.read_csv(data_path, low_memory=False)

print(f"Dataset shape: {customer_data.shape}")
print("Columns:", customer_data.columns.tolist())

# 1. Prepare data
customer_data['Date received'] = pd.to_datetime(customer_data['Date received'], errors='coerce')

# Check missing values and percentage
missing = customer_data.isnull().sum().sort_values(ascending=False)
pct_missing = (missing / len(customer_data) * 100).round(2)

# Top categories
top_products = customer_data['Product'].value_counts().head(10)
top_issues = customer_data['Issue'].value_counts().head(10)
top_companies = customer_data['Company'].value_counts().head(10)
top_states = customer_data['State'].value_counts().head(10)

# Complaints over time (monthly)
complaints_over_time = customer_data.groupby(customer_data['Date received'].dt.to_period("M")).size()

# Top 10 companies with percentage of total complaints
top_companies_df = (
    customer_data.groupby('Company')['Product']
    .count()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_companies_df["Pct"] = top_companies_df['Product'] / top_companies_df['Product'].sum() * 100
print(top_companies_df)

# === Plot 1: Missing values + Product distribution ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

pct_missing[pct_missing > 0].head(10).plot(kind='barh', ax=axes[0], color='tomato')
axes[0].set_title("Top 10 Columns by % Missing")
axes[0].invert_yaxis()

top_products.plot(kind='bar', ax=axes[1], color='steelblue')
axes[1].set_title("Top Products (by complaint count)")
axes[1].set_ylabel("Count")

plt.suptitle("Step 1: Data Quality & Target Overview", fontsize=14)
plt.tight_layout()
plt.show()

# === Plot 2: Issues and Companies ===
fig, axes = plt.subplots(1, 2, figsize=(15, 12))

top_issues.plot(kind='barh', ax=axes[0], color='purple')
axes[0].set_title("Top Issues (overall)")

top_companies.plot(kind='bar', ax=axes[1], color='darkorange')
axes[1].set_title("Top Companies by Volume")
axes[1].set_ylabel("Count")

plt.suptitle("Step 2: Nature of Complaints", fontsize=14)
plt.tight_layout()
plt.show()

# === Plot 3: States and Trends ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

top_states.plot(kind='bar', ax=axes[0], color='green')
axes[0].set_title("Top States by Complaints")

complaints_over_time.plot(ax=axes[1], color='red')
axes[1].set_title("Complaints Over Time (Monthly)")

plt.suptitle("Step 3: Geography & Time", fontsize=14)
plt.tight_layout()
plt.show()

# === Pie chart: Consumer disputed? proportions (if column exists) ===
if 'Consumer disputed?' in customer_data.columns:
    plt.figure(figsize=(5, 5))
    customer_data['Consumer disputed?'].fillna('Unknown').value_counts().plot(
        kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title("Consumer disputed? (proportions)")
    plt.ylabel("")
    plt.show()

'''
Narration:
- Show missingness — note Consumer complaint narrative high missingness (~70%). 
- For segmentation checks, use rows with Product present.
- Show product & issue distributions to orient stakeholders.
- Show top companies and states to reveal concentration and hotspots; time series to catch spikes (e.g., COVID).
'''

# 2. Prepare segmentation dataset (keep Product & segmentation columns)
seg_cols = ['Product', 'Issue', 'Company', 'State', 'Date received']
seg_df = customer_data[seg_cols].copy()

# Drop rows without Product (target required)
seg_df = seg_df.dropna(subset=['Product']).copy()

# Fill missing segment columns with 'Unknown' so categories are included
seg_df['Company'] = seg_df['Company'].fillna('Unknown')
seg_df['State'] = seg_df['State'].fillna('Unknown')

# Extract year from 'Date received'
seg_df['Year'] = pd.to_datetime(seg_df['Date received'], errors='coerce').dt.year.fillna(-1).astype(int)

print(f"Segmentation DataFrame shape: {seg_df.shape}")
print(f"Unique products: {seg_df['Product'].nunique()}")

'''
Exploratory check: company 'HSBC' products distribution (optional)
'''

# ==========================================================================================
# 3. Statistical tests with interpretation using Cramér's V and chi-squared test
# ==========================================================================================

def cramers_v_with_p(contingency_table):
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    r, k = contingency_table.shape
    v = np.sqrt((chi2 / n) / (min(r - 1, k - 1)))
    return v, p, chi2, dof

# Candidate categorical features to check association with Product
candidates = {
    'Company': seg_df['Company'],
    'State': seg_df['State'],
    'Year': seg_df['Year']
}

results = {}

for name, col in candidates.items():
    # Limit categories for large cardinality columns
    if name == 'Company':
        top_n = 30
        top_categories = seg_df['Company'].value_counts().head(top_n).index
        subset = seg_df[seg_df['Company'].isin(top_categories)]
        contingency_tab = pd.crosstab(subset['Company'], subset['Product'])
    elif name == 'State':
        top_n = 20
        top_categories = seg_df['State'].value_counts().head(top_n).index
        subset = seg_df[seg_df['State'].isin(top_categories)]
        contingency_tab = pd.crosstab(subset['State'], subset['Product'])
    else:  # Year
        contingency_tab = pd.crosstab(seg_df['Year'], seg_df['Product'])

    # Run test
    v, p, chi2_stat, dof = cramers_v_with_p(contingency_tab)
    results[name] = {
        'cramers_v': v,
        'p_value': p,
        'chi2_stat': chi2_stat,
        'dof': dof,
        'total_rows': contingency_tab.values.sum()
    }

    # Print results with interpretation
    print(f"\n{name} vs Product → Cramer's V = {v:.3f} | p-value = {p:.3e} | Total rows = {contingency_tab.values.sum()}")
    if p < 0.05 and v >= 0.3:
        print(f"Conclusion → Strong evidence: segmentation by {name.upper()} is justified.")
    elif p < 0.05 and v >= 0.1:
        print(f"Conclusion → Some evidence: segmentation by {name.upper()} may help.")
    else:
        print(f"Conclusion → Weak or no evidence: segmentation by {name.upper()} not needed.")

'''
Summary:

- Strong association between Company and Product found.
- Products are unevenly distributed across Companies.
- To avoid leakage, modeling will be global on narrative text, without segmenting by Company.
'''

# View results summary as DataFrame
results_df = pd.DataFrame(results).T
print("\nStatistical Test Results Summary:")
print(results_df)
