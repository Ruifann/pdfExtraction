import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load and clean the data
table_3 = pd.read_csv('ACHA_Tables/Table_3.csv')
table_3.to_csv('ACHA_Tables/cleaned_table3.csv', index=False)
table_3 = pd.read_csv('ACHA_Tables/cleaned_table3.csv')
table_3 = table_3.drop(columns=['Unnamed: 12'])
table_3.columns = ['Dimension', 'All Students', 'Cis Woman', 'Cis Man', 'Trans / GNC', 'BIPOC', 'Parent/ Guardian', 'Veterans', '1st Gen. College Students', 'Varsity Athletes', 'Disability/ Condition', 'Queer-spectrum', 'Visa']
indices_to_drop = list(range(0, 11)) + list(range(12, 15)) + [28]
table_3 = table_3.drop(indices_to_drop)

higher_the_better = ['Happiness', 'Life Satisfaction', 'Self-esteem', 'Optimism', 'Positive Coping', 'Belonging', 'Meaning', 'Purpose', 'Activity Engagement', 'Academic Engagement']
lower_the_better = ['Anxiety', 'Depression', 'Loneliness', 'Social Anxiety']

table_3['Dimension'] = table_3['Dimension'].replace({'life satisfaction': 'Life Satisfaction', 'social Anxiety': 'Social Anxiety'})


df_higher = table_3[table_3['Dimension'].isin(higher_the_better)]
df_lower = table_3[table_3['Dimension'].isin(lower_the_better)]


fig, ax = plt.subplots(figsize=(14, 8))
df_higher.set_index('Dimension').plot(kind='line', marker='o', ax=ax)
ax.set_title('Scores Across Different Happiness Dimensions for Student Subpopulations')
ax.set_xlabel('Dimension')
ax.set_ylabel('Score')
ax.legend(title='Subpopulations', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)  # Rotate dimension labels for better readability
plt.tight_layout()
plt.show()
fig.savefig(f'ACHA_Figures/Student_Population_Happiness.png')
plt.close()

fig, ax = plt.subplots(figsize=(14, 8))
df_lower.set_index('Dimension').plot(kind='line', marker='o', ax=ax)
ax.set_title('Scores Across Different Anxiety Dimensions for Student Subpopulations')
ax.set_xlabel('Dimension')
ax.set_ylabel('Score')
ax.legend(title='Subpopulations', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)  # Rotate dimension labels for better readability
plt.tight_layout()
plt.show()
fig.savefig(f'ACHA_Figures/Student_Population_Anxiety.png')
plt.close()
