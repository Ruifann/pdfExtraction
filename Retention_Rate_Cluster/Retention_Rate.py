import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
import os
import numpy as np
import textwrap

# Load the dataset
file_path = 'CSV_Retention_Rate_by_Race.csv'
data = pd.read_csv(file_path)

# Display the first few rows and columns of the data to understand its structure
print(data.head())
print(data.columns)

# Get the number of null values in each column
null_values = data.isnull().sum()

# Display the null values count
print("Null values in each column:")
print(null_values)

# Filter the data to include only four-year schools
four_year_schools = data[data['HD2022.Institutional category'] == 'Degree-granting, primarily baccalaureate or above']

# Columns of interest based on the dataset structure
columns_of_interest = [
    'institution name',
    'EF2022D.Full-time retention rate, 2022',
    'DRVGR2022.Graduation rate, American Indian or Alaska Native',
    'DRVGR2022.Graduation rate, Asian',
    'DRVGR2022.Graduation rate, Black, non-Hispanic',
    'DRVGR2022.Graduation rate, Hispanic',
    'DRVGR2022.Graduation rate, Native Hawaiian or Other Pacific Islander',
    'DRVGR2022.Graduation rate, White, non-Hispanic',
    'DRVGR2022.Graduation rate, two or more races',
    'DRVGR2022.Graduation rate, Race/ethnicity unknown',
    'DRVGR2022.Graduation rate, U.S. Nonresident'
]

# Select relevant columns
retention_data = four_year_schools[columns_of_interest].drop_duplicates()

# Fill null values with zero, assuming that null means no students in that category
retention_data = retention_data.fillna(0)

# Remove prefixes from column names for better readability
renamed_columns = {
    'EF2022D.Full-time retention rate, 2022': 'Full-time retention rate, 2022',
    'DRVGR2022.Graduation rate, American Indian or Alaska Native': 'American Indian or Alaska Native',
    'DRVGR2022.Graduation rate, Asian': 'Asian',
    'DRVGR2022.Graduation rate, Black, non-Hispanic': 'Black, non-Hispanic',
    'DRVGR2022.Graduation rate, Hispanic': 'Hispanic',
    'DRVGR2022.Graduation rate, Native Hawaiian or Other Pacific Islander': 'Native Hawaiian or Other Pacific Islander',
    'DRVGR2022.Graduation rate, White, non-Hispanic': 'White, non-Hispanic',
    'DRVGR2022.Graduation rate, two or more races': 'two or more races',
    'DRVGR2022.Graduation rate, Race/ethnicity unknown': 'Race/ethnicity unknown',
    'DRVGR2022.Graduation rate, U.S. Nonresident': 'U.S. Nonresident'
}

retention_data.rename(columns=renamed_columns, inplace=True)
#
# # Display the processed data
# print(retention_data.head())
#
# # Standardize the data (excluding the first column)
# scaler = StandardScaler()
# normalized_data = scaler.fit_transform(retention_data.iloc[:, 1:])
#
# # Perform KMeans clustering
# kmeans = KMeans(n_clusters=6, random_state=42)
# clusters = kmeans.fit_predict(normalized_data)
# retention_data['Cluster'] = clusters
#
# # Apply PCA for visualization
# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(normalized_data)
#
# # Prepare the DataFrame for plotting
# pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
# pca_df['Cluster'] = clusters
# pca_df['Institution'] = retention_data['institution name']
#
# # Function to extract capitals from institution name
# def extract_capitals(name):
#     if isinstance(name, str):
#         return ''.join(re.findall(r'[A-Z]', name))
#     return ''
#
# # Convert the Institution column to string type and apply the extract_capitals function
# pca_df['Institution'] = pca_df['Institution'].astype(str).apply(extract_capitals)
#
# # Plot the clusters
# plt.figure(figsize=(12, 8))
# sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
# for line in range(0, pca_df.shape[0]):
#     plt.text(pca_df.PC1[line], pca_df.PC2[line], pca_df.Institution[line], horizontalalignment='left', size='small', color='black', alpha=0.6)
# plt.title('University Clustering based on Retention Rate by Race')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.legend(title='Cluster')
# plt.grid(True)
# plt.savefig('university_clusters_retention_rate.png')
# plt.show()
#
# # Create directories for output if they don't exist
# heatmap_dir = 'Clustered_University_Heatmap_Retention_Chart'
# os.makedirs(heatmap_dir, exist_ok=True)
#
# # Function to wrap text for long labels
# def wrap_labels(ax, width=10):
#     labels = []
#     for label in ax.get_xticklabels():
#         text = label.get_text()
#         wrapped_text = '\n'.join(textwrap.wrap(text, width))
#         labels.append(wrapped_text)
#     ax.set_xticklabels(labels, rotation=45, ha='right')
#
# # Create heatmaps for universities in the same cluster
# clusters = retention_data['Cluster'].unique()
# race_labels = [
#     'American Indian or Alaska Native',
#     'Asian',
#     'Black, non-Hispanic',
#     'Hispanic',
#     'Native Hawaiian or Other Pacific Islander',
#     'White, non-Hispanic',
#     'two or more races',
#     'Race/ethnicity unknown',
#     'U.S. Nonresident'
# ]
#
# for cluster in clusters:
#     cluster_data = retention_data[retention_data['Cluster'] == cluster]
#     cluster_data = cluster_data.set_index('institution name')
#
#     plt.figure(figsize=(12, 8))
#     ax = sns.heatmap(cluster_data[race_labels], annot=True, cmap='viridis', fmt='g')
#     plt.title(f'Graduation Rate Diversity for Cluster {cluster}')
#     plt.ylabel('Institution Name')
#     plt.xlabel('Graduation Rate Category')
#     wrap_labels(ax, 15)
#     plt.tight_layout()
#     plt.savefig(os.path.join(heatmap_dir, f'Cluster_{cluster}_heatmap_chart.png'), bbox_inches="tight")
#     plt.show()
Ohio = data[retention_data['institution name'] == 'Ohio State University-Main Campus']
print(Ohio)