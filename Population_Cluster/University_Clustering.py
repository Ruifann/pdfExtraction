import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re
import os

file_path = 'CSV_Level_of_Student_by_Race.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the data to understand its structure
data.head()

# Filter the data to include only rows where the student level is "All students total"
filtered_data = data[data['EF2022A.Level of student'] == 'All students total']

# Drop rows with missing institution names
filtered_data = filtered_data.dropna(subset=['institution name'])

# Corrected columns of interest for the filtered data
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
    'EF2022A.U.S. Nonresident total'
]

diversity_data = filtered_data[columns_of_interest].drop_duplicates()
diversity_data = diversity_data.fillna(0)

scaler = StandardScaler()
normalized_data = scaler.fit_transform(diversity_data.iloc[:, 1:])

kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(normalized_data)
diversity_data['Cluster'] = clusters

# Apply PCA for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(normalized_data)

# Ensure correct alignment by resetting index before PCA transformation
diversity_data = diversity_data.reset_index(drop=True)

pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Cluster'] = clusters
pca_df['Institution'] = diversity_data['institution name']


def extract_capitals(name):
    return ''.join(re.findall(r'[A-Z]', name))


pca_df['Institution'] = pca_df['Institution'].apply(extract_capitals)

# Plot the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
for line in range(0, pca_df.shape[0]):
    plt.text(pca_df.PC1[line], pca_df.PC2[line], pca_df.Institution[line], horizontalalignment='left', size='small', color='black', alpha=0.6)
plt.title('University Clustering based on Population Diversity')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.savefig('university_clusters_combined.png')
plt.show()

# Display the resulting DataFrame
print(diversity_data.info())

output_dir = 'University_Population_Pie_Charts'
os.makedirs(output_dir, exist_ok=True)


# Filter the data to include only rows where the student level is "All students total"
filtered_data = data[data['EF2022A.Level of student'] == 'All students total']

# Drop rows with missing institution names
filtered_data = filtered_data.dropna(subset=['institution name'])

# Corrected columns of interest for the filtered data
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
    'EF2022A.U.S. Nonresident total'
]

diversity_data = filtered_data[columns_of_interest].drop_duplicates()
diversity_data = diversity_data.fillna(0)


# Function to create and save pie chart
def create_pie_chart(row):
    labels = [
        'American Indian or Alaska Native', 'Asian', 'Black or African American',
        'Hispanic', 'Native Hawaiian or Other Pacific Islander', 'White',
        'Two or more races', 'Race/ethnicity unknown', 'U.S. Nonresident'
    ]
    sizes = row[1:].values
    institution_name = row['institution name']

    # Filter out zero values
    sizes = sizes[sizes > 0]
    labels = [label for label, size in zip(labels, row[1:].values) if size > 0]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title(f'Population Diversity of {institution_name}')
    plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(output_dir, f'{institution_name}_pie_chart.png'), bbox_inches="tight")
    plt.close()


# Generate pie charts for all institutions
for index, row in diversity_data.iterrows():
    create_pie_chart(row)
