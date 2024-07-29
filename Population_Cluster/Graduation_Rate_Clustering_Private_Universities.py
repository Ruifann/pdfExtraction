import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import re

# Load the datasets
file_path_grad = 'CSV_Graduation_Rate_All_Universities.csv'
grad_data = pd.read_csv(file_path_grad)

file_path_div = 'Updated_Population_Diversity_with_Clusters.csv'
div_data = pd.read_csv(file_path_div)

# Merge the datasets
merged_data = pd.merge(grad_data, div_data, on='institution name', how='inner')

# Filter private schools
private_school_data = merged_data[merged_data['HD2022.Sector of institution'] == 'Private not-for-profit, 4-year or above']
private_school_data = private_school_data.dropna(subset=['DRVGR2022.Graduation rate, total cohort'])

# Filter rows where EF2022A.Grand total is greater than 1000 and less than 60000
private_school_data = private_school_data[(private_school_data['EF2022A.Grand total'] > 1000) & (private_school_data['EF2022A.Grand total'] < 60000)]

# Columns of interest based on the dataset structure
columns_of_interest = [
    'institution name',
    'DRVGR2022.Graduation rate, total cohort',
    'HD2022.Degree of urbanization (Urban-centric locale)',
    'EF2022D.Full-time retention rate, 2022',
    'American Indian or Alaska Native',
    'Asian',
    'Black or African American',
    'Hispanic',
    'Native Hawaiian or Other Pacific Islander',
    'White',
    'Two or more races',
    'Race/ethnicity unknown',
    'U.S. Nonresident',
    'EF2022A.Grand total',
]

# Select relevant columns
df_selected = private_school_data[columns_of_interest]
print(df_selected.info())
df_selected.to_csv('private_schools_new_clusters.csv', index=False)

# Load the processed dataset
file_path = 'private_schools_new_clusters.csv'
data = pd.read_csv(file_path)

institution_names = data['institution name']
df_selected = data.drop(columns=['institution name'])

# Apply one-hot encoding to categorical columns
categorical_columns = ['HD2022.Degree of urbanization (Urban-centric locale)']
df_encoded = pd.get_dummies(df_selected, columns=categorical_columns)

# Handle missing values and filter rows based on EF2022A.Grand total
df_encoded.fillna(0, inplace=True)
df_encoded = df_encoded[(df_encoded['EF2022A.Grand total'] > 1000) & (df_encoded['EF2022A.Grand total'] < 60000)]

# Apply Min-Max Scaling to EF2022A.Grand total
scaler = RobustScaler()
df_encoded['EF2022A.Grand total'] = scaler.fit_transform(df_encoded[['EF2022A.Grand total']])

# Compute the correlation matrix
corr_matrix = df_encoded.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

# Find features with correlation greater than a threshold
threshold = 0.9
highly_correlated_features = [column for column in upper.columns if any(upper[column] > threshold)]

# Drop highly correlated features
df_encoded_dropped = df_encoded.drop(columns=highly_correlated_features)

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded_dropped)

# Fit PCA
pca = PCA(n_components=2)
pca.fit(df_scaled)

# Get the loadings (coefficients of the features)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Create a DataFrame for the loadings
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=df_encoded_dropped.columns)

# Plot the loadings
plt.figure(figsize=(20, 14))
plt.bar(loadings_df.index, loadings_df['PC1'], alpha=0.5, align='center', label='PC1')
plt.bar(loadings_df.index, loadings_df['PC2'], alpha=0.5, align='center', label='PC2')
plt.ylabel('Loading Scores')
plt.xlabel('Features')
plt.title('Feature Contributions to Principal Components')
plt.xticks(rotation=90)
plt.legend()
plt.show()

# Print the loadings DataFrame for detailed examination
print(loadings_df)

# Fit the KMeans model with the chosen number of clusters
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_selected = df_selected.copy()  # Make a copy to avoid SettingWithCopyWarning
df_selected.loc[:, 'KMeans_Cluster'] = kmeans.fit_predict(df_encoded_dropped)

# Calculate performance metrics for KMeans
kmeans_silhouette_avg = silhouette_score(df_encoded_dropped, df_selected['KMeans_Cluster'])
kmeans_davies_bouldin_avg = davies_bouldin_score(df_encoded_dropped, df_selected['KMeans_Cluster'])

print(f"KMeans Silhouette Score after removing highly correlated features: {kmeans_silhouette_avg}")
print(f"KMeans Davies-Bouldin Index after removing highly correlated features: {kmeans_davies_bouldin_avg}")

# Apply PCA for visualization in 2D
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_encoded_dropped)

# Prepare the DataFrame for plotting
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['KMeans_Cluster'] = df_selected['KMeans_Cluster']
pca_df['Institution'] = data['institution name']

# Function to extract capitals from institution name
def extract_capitals(name):
    if isinstance(name, str):
        return ''.join(re.findall(r'[A-Z]', name))
    return ''

pca_df['Institution'] = pca_df['Institution'].apply(extract_capitals)

# Plot the KMeans clusters in 2D
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='KMeans_Cluster', palette='viridis', s=100, alpha=0.7)
for line in range(0, pca_df.shape[0]):
    plt.text(pca_df.PC1[line], pca_df.PC2[line], pca_df.Institution[line], horizontalalignment='left', size='small', color='black', alpha=0.6)
plt.title('University Clustering based on Graduation Rates and Additional Features (KMeans, 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='KMeans_Cluster')
plt.grid(True)
plt.tight_layout()
plt.savefig('private_university_clusters_combined_kmeans_2d_no_scaling.png')
plt.show()

df_selected['institution name'] = institution_names
div_data_selected = div_data[['institution name', 'Cluster']]
merged_data = pd.merge(df_selected, div_data_selected, on='institution name', how='inner')
merged_data.to_csv('private_schools_clusters_no_scaling.csv', index=False)
