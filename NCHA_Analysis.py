import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

table_21 = pd.read_csv('NCHA_Tables/Table_21.csv', header=None)
table_21.columns = ['Impediments', 'Cis_Men', 'Cis_Women', 'Trans/GNC', 'Total', 'Cis_Men_Pop', 'Cis_Women_Pop', 'Trans/GNC_Pop', 'Total_Pop']
table_21.set_index('Impediments', inplace=True)
table_21 = table_21.sort_values(by='Total', ascending=True)
table_21.index = table_21.index.to_series().replace({
    'Eating disorder/problem': 'Eating disorder',
    'Headaches/migraines': 'Headaches',
    'Influenza or influenza-like illness (the flu)': 'Influenza'
})
# Create a figure and a set of subplots
fig, ax = plt.subplots()

fig.set_size_inches(12, 6)
# Stacked bar chart
table_21[['Cis_Men', 'Cis_Women', 'Trans/GNC']].plot(kind='bar', ax=ax, color=['#FFA500', '#008080', '#D3D3D3'], width=0.6)

# Line chart
table_21['Total'].plot(kind='line', marker='o', color='red', ax=ax, linewidth=1.2)


# Titles and labels
ax.set_ylabel('Total Counts')
plt.title('Comparison of Impediments to Academic Performance by Gender')

ax.set_xticklabels(table_21.index, rotation=45, ha="right")

ax.legend(['Total', 'Cis_Men', 'Cis_Women', 'Trans/GNC'], title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

# Show plot
plt.show()
fig.savefig(f'NCHA_Figures/Impediments_to_Academic_Performance_by_Gender.png')
plt.close()

Table_68 = pd.read_csv('NCHA_Tables/Table_68.csv', header=None)

# Extract category and percentage
Table_68[['Category', 'Percentage']] = Table_68[0].str.extract(r'^(.*?)(\d+\.\d+ %)$')

# Clean the data
Table_68['Category'] = Table_68['Category'].str.strip()
Table_68['Percentage'] = Table_68['Percentage'].str.replace(' %', '').astype(float)

# Drop the original column
Table_68.drop(columns=[0], inplace=True)

# Show the result
print(Table_68)

table_4 = pd.read_csv('USC_CDS_2022_Tables/Table_4.csv', skiprows=6)
# Drop unnecessary columns and clean up the DataFrame
table_4_cleaned = table_4[['Unnamed: 0', 'seeking)']].copy()
table_4_cleaned.columns = ['Category', 'Number']
table_4_cleaned['Number'] = pd.to_numeric(table_4_cleaned['Number'].str.replace(',', ''), errors='coerce')


# Standardize category names by extracting common terms
def standardize_category(name):
    name = name.lower()
    if 'black' in name:
        return 'Black or African American'
    elif 'white' in name:
        return 'White'
    elif 'latino' in name:
        return 'Hispanic or Latino/a/x'
    elif 'asian' in name:
        return 'Asian or Asian American'
    elif 'two or more' in name or 'biracial' in name:
        return 'Biracial or Multiracial'
    elif 'american indian' in name:
        return 'American Indian or Native Alaskan'
    elif 'native hawaiian' in name:
        return 'Native Hawaiian or Other Pacific Islander'
    elif 'unknown' in name or 'identity' in name:
        return 'Identity not listed above'
    return name


table_4_cleaned['Category'] = table_4_cleaned['Category'].apply(standardize_category)
table_4_cleaned = table_4_cleaned[table_4_cleaned['Category'] != 'total']
Table_68['Category'] = Table_68['Category'].apply(standardize_category)
Table_68 = Table_68.sort_values(by='Category')
table_4_cleaned = table_4_cleaned.sort_values(by='Category')
# Color mapping based on standardized category names
color_map = {
    'Black or African American': '#ff9999',
    'White': '#66b3ff',
    'Hispanic or Latino/a/x': '#99ff99',
    'Asian or Asian American': '#ffcc99',
    'Middle Eastern/North African (MENA) or Arab Origin': '#b15928',
    'Biracial or Multiracial': '#c2c2f0',
    'American Indian or Native Alaskan': '#ffb3e6',
    'Native Hawaiian or Other Pacific Islander': '#c4e17f',
    'Identity not listed above': '#f7c6e8',
    'nonresidents': '#d4e157'  # Unique color for nonresident
}

# Create the adjusted pie charts with external labels and a legend
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Pie chart for Table_68 mockup
wedges, texts, autotexts = ax[0].pie(Table_68['Percentage'], autopct='%1.1f%%',
                                     colors=[color_map.get(x, '#d9d9d9') for x in Table_68['Category']],
                                     textprops=dict(color="black"), pctdistance=0.85)

# Draw circle to make it look like a donut
ax[0].set(aspect="equal", title='Pie Chart for NCHA Survey Student Population')
ax[0].legend(wedges, Table_68['Category'], title="Racial/Ethnic Category", loc="center left", bbox_to_anchor=(0.9, 0, 0.5, 1))

# Pie chart for Table_4
wedges2, texts2, autotexts2 = ax[1].pie(table_4_cleaned['Number'], autopct='%1.1f%%',
                                        colors=[color_map.get(x, '#d9d9d9') for x in table_4_cleaned['Category']],
                                        textprops=dict(color="black"), pctdistance=0.85)
# Draw circle to make it look like a donut
ax[1].set(aspect="equal", title='Pie Chart for USC Student Population')
ax[1].legend(wedges2, table_4_cleaned['Category'], title="Racial/Ethnic Category", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()
plt.show()
fig.savefig(f'NCHA_Figures/NCHA_and_USC_Student_Population_Comparison.png')
plt.close()

