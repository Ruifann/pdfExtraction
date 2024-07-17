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
file_path = 'CSV_Graduation_Rate_All_Schools.csv'
data = pd.read_csv(file_path)

print(data.head())

# Filter public and private schools
public_school_data = data[data['HD2022.Sector of institution'] == 'Public, 4-year or above']
public_school_data = public_school_data.dropna(subset=['DRVGR2022.Graduation rate, women'])
public_school_data = public_school_data[public_school_data['DRVGR2022.Graduation rate, total cohort'] != 0]
print(public_school_data.info())

private_school_data = data[data['HD2022.Sector of institution'] == 'Private not-for-profit, 4-year or above']
private_school_data = private_school_data.dropna(subset=['DRVGR2022.Graduation rate, total cohort'])
private_school_data = private_school_data[private_school_data['DRVGR2022.Graduation rate, total cohort'] != 0]
print(private_school_data.info())

# Columns of interest based on the dataset structure
columns_of_interest = [
    'institution name',
    'DRVGR2022.Graduation rate, American Indian or Alaska Native',
    'DRVGR2022.Graduation rate, Asian/Native Hawaiian/Other Pacific Islander',
    'DRVGR2022.Graduation rate, Asian',
    'DRVGR2022.Graduation rate, Native Hawaiian or Other Pacific Islander',
    'DRVGR2022.Graduation rate, Black, non-Hispanic',
    'DRVGR2022.Graduation rate, Hispanic',
    'DRVGR2022.Graduation rate, White, non-Hispanic',
    'DRVGR2022.Graduation rate, two or more races',
    'DRVGR2022.Graduation rate, Race/ethnicity unknown',
    'DRVGR2022.Graduation rate, U.S. Nonresident',
    'HD2022.Degree of urbanization (Urban-centric locale)',
    'HD2022.Institution size category'
]

# Select relevant columns
df_selected = public_school_data[columns_of_interest]

# Apply one-hot encoding to categorical columns
df_encoded = pd.get_dummies(df_selected, columns=[
    'HD2022.Degree of urbanization (Urban-centric locale)',
    'HD2022.Institution size category'
])

# Drop the 'institution name' column since it is not needed for clustering
df_encoded = df_encoded.drop(columns=['institution name'])

# Handle missing values (e.g., fill with mean)
df_encoded.fillna(df_encoded.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Determine the optimal number of clusters using the Elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    sse.append(kmeans.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Choose the optimal number of clusters (e.g., 3 based on the elbow plot)
optimal_clusters = 3

# Fit the KMeans model with the chosen number of clusters
df_selected = df_selected.copy()  # Ensure a copy is used to avoid SettingWithCopyWarning
df_selected['Cluster'] = kmeans.fit_predict(df_scaled)

# Apply PCA for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)

# Prepare the DataFrame for plotting
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df_selected['Cluster']
pca_df['Institution'] = df_selected['institution name']

# Function to extract capitals from institution name
def extract_capitals(name):
    if isinstance(name, str):
        return ''.join(re.findall(r'[A-Z]', name))
    return ''

pca_df['Institution'] = pca_df['Institution'].apply(extract_capitals)

# Plot the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
for line in range(0, pca_df.shape[0]):
    plt.text(pca_df.PC1[line], pca_df.PC2[line], pca_df.Institution[line], horizontalalignment='left', size='small', color='black', alpha=0.6)
plt.title('University Clustering based on Graduation Rates and Additional Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('university_clusters_combined.png')
plt.show()

# Display the resulting DataFrame
print(df_selected.info())

# Get the list of schools in each cluster
clusters = df_selected.groupby('Cluster')['institution name'].apply(list)

# Print the schools in each cluster
for cluster, schools in clusters.items():
    print(f"Cluster {cluster}:")
    for school in schools:
        print(f"  - {school}")
df_selected.to_csv('public_schools_clusters.csv', index=False)