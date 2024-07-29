import pandas as pd

# Load the datasets
public_schools_data_path = 'public_schools_new_clusters_no_scaling.csv'
private_schools_data_path = 'private_schools_clusters_no_scaling.csv'
public_schools_data = pd.read_csv(public_schools_data_path)
private_schools_data = pd.read_csv(private_schools_data_path)


def get_schools_in_same_cluster(school_name, data):
    cluster_number = data[data['institution name'] == school_name]['Cluster'].values[0]
    kmeans_cluster_number = data[data['institution name'] == school_name]['KMeans_Cluster'].values[0]
    schools_in_same_cluster = data[(data['Cluster'] == cluster_number) & (data['KMeans_Cluster'] == kmeans_cluster_number)]['institution name'].tolist()
    return schools_in_same_cluster


def is_public_school(school_name, public_data, private_data):
    if school_name in public_data['institution name'].values:
        return True
    elif school_name in private_data['institution name'].values:
        return False
    else:
        raise ValueError(f"School name '{school_name}' not found in either dataset.")


# Example school names
school_name = 'University of Southern California'
school_names = ['Idaho State University', 'Illinois State University']

# Determine if the school is public or private
if is_public_school(school_name, public_schools_data, private_schools_data):
    clustered_data = public_schools_data
    print(f"{school_name} is a public school.")
else:
    clustered_data = private_schools_data
    print(f"{school_name} is a private school.")

# Get schools in the same cluster
schools_in_same_cluster = get_schools_in_same_cluster(school_name, clustered_data)
print(f"Schools in the same cluster as {school_name}:")
for school in schools_in_same_cluster:
    print(f"  - {school}")

# Print data for other example school names
for name in school_names:
    if is_public_school(name, public_schools_data, private_schools_data):
        school_data = public_schools_data[public_schools_data['institution name'] == name]
    else:
        school_data = private_schools_data[private_schools_data['institution name'] == name]

    for index, row in school_data.iterrows():
        print(f"Data for {row['institution name']}:")
        for column in school_data.columns:
            print(f"{column}: {row[column]}")
        print("\n")
