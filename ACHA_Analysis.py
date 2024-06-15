import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

table_3 = pd.read_csv('ACHA_Tables/Table_3.csv')
table_3.to_csv('ACHA_Tables/cleaned_table3.csv', index=False)
table_3 = pd.read_csv('ACHA_Tables/cleaned_table3.csv')
table_3 = table_3.drop(columns=['Unnamed: 12'])
table_3.columns = ['Dimension', 'All Students', 'Cis Woman', 'Cis Man', 'Trans / GNC', 'BIPOC', 'Parent/ Guardian', 'Veterans', '1st Gen. College Students', 'Varsity Athletes', 'Disability/ Condition', 'Queer-spectrum', 'Visa']
indices_to_drop = list(range(0, 11)) + list(range(12, 15)) + [28]
table_3 = table_3.drop(indices_to_drop)
print(table_3)
table_3.to_csv('ACHA_Tables/cleaned_table3.csv')

# Classify dimensions
higher_the_better = ['Happiness', 'Life Satisfaction', 'Self-esteem', 'Optimism', 'Positive Coping', 'Belonging', 'Meaning', 'Purpose', 'Activity Engagement', 'Academic Engagement']
lower_the_better = ['Anxiety', 'Depression', 'Loneliness', 'Social Anxiety']

# Filter data for each category
df_higher = table_3[table_3['Dimension'].isin(higher_the_better)]
df_lower = table_3[table_3['Dimension'].isin(lower_the_better)]

print(df_lower)


fig, ax = plt.subplots(figsize=(14, 8))
for col in df_higher.columns[1:]:  # Exclude the 'Dimension' column for plotting
    ax.plot(df_higher['Dimension'], df_higher[col], marker='o', label=col)
ax.set_ylim(25, 55)
ax.set_title('Scores Across Different Happiness Dimensions for Student Subpopulations')
ax.set_xlabel('Dimension')
ax.set_ylabel('Score')
ax.set_yticks(np.array([25, 30, 35, 40, 45, 50, 55]))
ax.set_yticklabels([25, 30, 35, 40, 45, 50, 55])
ax.legend(title='Subpopulations', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
fig.savefig(f'ACHA_Figures/Student_Population_Happiness.png')
plt.close()

fig, ax = plt.subplots(figsize=(14, 8))
for col in df_lower.columns[1:]:  # Exclude the 'Dimension' column for plotting
    ax.plot(df_lower['Dimension'], df_lower[col], marker='o', label=col)
ax.set_ylim(15, 35)
ax.set_title('Scores Across Different Anxiety Dimensions for Student Subpopulations')
ax.set_xlabel('Dimension')
ax.set_ylabel('Score')
ax.set_yticks(np.array([15, 20, 25, 30, 35]))
ax.set_yticklabels([15, 20, 25, 30, 35])
ax.legend(title='Subpopulations', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
fig.savefig(f'ACHA_Figures/Student_Population_Anxiety.png')
plt.close()
