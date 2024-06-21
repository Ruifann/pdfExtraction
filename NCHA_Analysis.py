import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
