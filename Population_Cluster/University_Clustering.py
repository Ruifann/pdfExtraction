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
file_path = 'CSV_Level_of_Student_by_Race.csv'
data = pd.read_csv(file_path)

# Display the first few rows and columns of the data to understand its structure
print(data.head())
print(data.columns)

# Filter the data to include only rows where the student level is "All students total"
filtered_data = data[data['EF2022A.Level of student'] == 'All students total']

# Drop rows with missing institution names
filtered_data = filtered_data.dropna(subset=['institution name'])

# Columns of interest based on the dataset structure
columns_of_interest = [
    'institution name',
    'EF2022A.American Indian or Alaska Native total',
    'EF2022A.Asian total',
    'EF2022A.Black or African American total',
    'EF2022A.Hispanic total',
    'EF2022A.Native Hawaiian or Other Pacific Islander total',
    'EF2022A.White total',
    'EF2022A.Two or more races total',
    'EF2022A.Race/ethnicity unknown total',
    'EF2022A.U.S. Nonresident total',
    'EF2022A.Grand total'
]

# Create a mapping for readable labels
readable_labels = {
    'EF2022A.American Indian or Alaska Native total': 'American Indian or Alaska Native',
    'EF2022A.Asian total': 'Asian',
    'EF2022A.Black or African American total': 'Black or African American',
    'EF2022A.Hispanic total': 'Hispanic',
    'EF2022A.Native Hawaiian or Other Pacific Islander total': 'Native Hawaiian or Other Pacific Islander',
    'EF2022A.White total': 'White',
    'EF2022A.Two or more races total': 'Two or more races',
    'EF2022A.Race/ethnicity unknown total': 'Race/ethnicity unknown',
    'EF2022A.U.S. Nonresident total': 'U.S. Nonresident'
}

# Replace the column names with readable labels for plotting
diversity_data = filtered_data[columns_of_interest].drop_duplicates()
diversity_data = diversity_data.fillna(0)
diversity_data.rename(columns=readable_labels, inplace=True)

# Calculate percentages
for column in readable_labels.values():
    diversity_data[column] = (diversity_data[column] / diversity_data['EF2022A.Grand total']) * 100

# Standardize the data (excluding the Grand total)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(diversity_data.iloc[:, 1:-1])


# Remove outliers using Z-score
z_scores = np.abs((normalized_data - np.mean(normalized_data, axis=0)) / np.std(normalized_data, axis=0))
threshold = 3
outliers = (z_scores > threshold).any(axis=1)
filtered_normalized_data = normalized_data[~outliers]
filtered_diversity_data = diversity_data.loc[~outliers].reset_index(drop=True)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(filtered_normalized_data)
filtered_diversity_data.loc[:, 'Cluster'] = clusters

# Apply PCA for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(filtered_normalized_data)

# Prepare the DataFrame for plotting
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters
pca_df['Institution'] = filtered_diversity_data['institution name']


# Function to extract capitals from institution name
def extract_capitals(name):
    return ''.join(re.findall(r'[A-Z]', name))


pca_df['Institution'] = pca_df['Institution'].apply(extract_capitals)

# Plot the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
for line in range(0, pca_df.shape[0]):
    plt.text(pca_df.PC1[line], pca_df.PC2[line], pca_df.Institution[line], horizontalalignment='left', size='small',
             color='black', alpha=0.6)
plt.title('University Clustering based on Population Diversity')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('university_clusters_combined.png')
plt.show()

# Display the resulting DataFrame
print(filtered_diversity_data.info())

# Create directories for output if they don't exist
pie_chart_dir = 'University_Population_Pie_Charts'
stacked_bar_dir = 'Clustered_University_Stack_Pie_Chart'
os.makedirs(pie_chart_dir, exist_ok=True)
os.makedirs(stacked_bar_dir, exist_ok=True)


# Function to create and save pie chart
def create_pie_chart(row):
    labels = [
        'American Indian or Alaska Native', 'Asian', 'Black or African American',
        'Hispanic', 'Native Hawaiian or Other Pacific Islander', 'White',
        'Two or more races', 'Race/ethnicity unknown', 'U.S. Nonresident'
    ]
    sizes = row[1:-2].values  # Exclude 'institution name', 'Cluster', and 'Grand total' columns
    institution_name = row['institution name']

    # Filter out zero values
    sizes = sizes[sizes > 0]
    labels = [label for label, size in zip(labels, row[1:-2].values) if size > 0]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f'Population Diversity of {institution_name}')
    plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(pie_chart_dir, f'{institution_name}_pie_chart.png'), bbox_inches="tight")
    plt.close()


# Generate pie charts for all institutions
for index, row in filtered_diversity_data.iterrows():
    create_pie_chart(row)


# Function to wrap text for long labels
def wrap_labels(ax, width=10):
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        labels.append('\n'.join(textwrap.wrap(text, width)))
    ax.set_yticklabels(labels, rotation=0, ha='right')


# Create horizontal stacked bar charts for universities in the same cluster
clusters = filtered_diversity_data['Cluster'].unique()
race_labels = [
    'American Indian or Alaska Native',
    'Asian',
    'Black or African American',
    'Hispanic',
    'Native Hawaiian or Other Pacific Islander',
    'White',
    'Two or more races',
    'Race/ethnicity unknown',
    'U.S. Nonresident'
]

for cluster in clusters:
    cluster_data = filtered_diversity_data[filtered_diversity_data['Cluster'] == cluster]
    cluster_data = cluster_data.set_index('institution name')

    # Dynamically set figure size based on the number of institutions
    num_institutions = len(cluster_data)
    fig_height = max(8, int(num_institutions * 0.5))
    plt.figure(figsize=(12, fig_height))
    ax = cluster_data[race_labels].plot(kind='barh', stacked=True, figsize=(12, fig_height), width=0.8)
    plt.title(f'Student Population Diversity for Cluster {cluster}')
    plt.ylabel('Institution Name')
    plt.xlabel('Percentage of Students')
    plt.legend(title='Race/Ethnicity', bbox_to_anchor=(1.05, 1), loc='upper left')
    wrap_labels(ax, 15)  # Wrap labels to fit better
    plt.tight_layout()
    plt.savefig(os.path.join(stacked_bar_dir, f'Cluster_{cluster}_stacked_barh_chart.png'), bbox_inches="tight")
    plt.show()
