import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

def map_categorical(df, column, mapping_dict, fill_value='Unknown'):
    """
    Maps a categorical column in a DataFrame using the provided mapping dictionary.
    Fills unmapped or missing values with the specified fill_value.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column.
    - column (str): The name of the column to map.
    - mapping_dict (dict): The mapping dictionary with keys as strings.
    - fill_value (str): The value to fill for unmapped or missing entries.
    
    Returns:
    - pd.Series: The mapped column.
    """
    mapped_series = df[column].astype(str).map(mapping_dict)
    mapped_series.fillna(fill_value, inplace=True)
    return mapped_series

# 1. Read the participants.tsv file
participants_df = pd.read_csv('participants.tsv', sep='\t')

# 2. Load the participants.json file
with open('participants.json', 'r') as json_file:
    participants_meta = json.load(json_file)

# 3. Map 'Sex' codes to labels using the metadata (without inverting)
sex_mapping = participants_meta['Sex']['Levels']
participants_df['Sex'] = map_categorical(participants_df, 'Sex', sex_mapping)

# Debug: Verify 'Sex' mapping
print("\nSex Counts:")
print(participants_df['Sex'].value_counts())

# 4. Convert height and weight to metric units
participants_df['Height_m'] = participants_df['Height_W1'] * 0.0254
participants_df['Weight_kg'] = participants_df['Weight_W1'] * 0.453592

# 5. Calculate BMI
participants_df['BMI_Calculated'] = participants_df['Weight_kg'] / (participants_df['Height_m'] ** 2)

# 6. Plot age distribution and save as PNG with customizations
plt.figure(figsize=(10, 6))
plt.hist(participants_df['AgeMRI_W1'].dropna(), bins=20, edgecolor='black', color='skyblue')
plt.title('Age Distribution at Wave 1 MRI', fontsize=16)
plt.xlabel('Age (years)', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('age_distribution.png', dpi=300)
plt.close()

# 7. Plot sex distribution and save as PNG with customizations
plt.figure(figsize=(6, 6))
sex_counts = participants_df['Sex'].value_counts()
sex_counts.plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Sex Distribution of Participants', fontsize=16)
plt.xlabel('Sex', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('sex_distribution.png', dpi=300)
plt.close()

# 8. Calculate age differences between waves
participants_df['AgeDifference_W1_W2'] = participants_df['AgeMRI_W2'] - participants_df['AgeMRI_W1']

# 9. Filter participants with complete MRI data
complete_cases = participants_df.dropna(subset=['AgeMRI_W1', 'AgeMRI_W2', 'AgeMRI_W3'])
print(f"\nParticipants with MRI data at all waves: {len(complete_cases)}")

# 10. Save processed data
participants_df.to_csv('participants_processed.tsv', sep='\t', index=False)

# Additional Metrics and Visualizations

# 11. Plot BMI distribution at Wave 1
plt.figure(figsize=(10, 6))
plt.hist(participants_df['BMI_W1'].dropna(), bins=20, edgecolor='black', color='seagreen')
plt.title('BMI Distribution at Wave 1', fontsize=16)
plt.xlabel('BMI', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('bmi_distribution.png', dpi=300)
plt.close()

# 12. Plot MMSE scores at Wave 1, Wave 2, and Wave 3
# MMSE Wave 1
plt.figure(figsize=(10, 6))
plt.hist(participants_df['MMSE_W1'].dropna(), bins=15, edgecolor='black', color='cornflowerblue')
plt.title('MMSE Score Distribution at Wave 1', fontsize=16)
plt.xlabel('MMSE Score', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('mmse_w1_distribution.png', dpi=300)
plt.close()

# MMSE Wave 2
plt.figure(figsize=(10, 6))
plt.hist(participants_df['MMSE_W2'].dropna(), bins=15, edgecolor='black', color='salmon')
plt.title('MMSE Score Distribution at Wave 2', fontsize=16)
plt.xlabel('MMSE Score', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('mmse_w2_distribution.png', dpi=300)
plt.close()

# MMSE Wave 3
plt.figure(figsize=(10, 6))
plt.hist(participants_df['MMSE_W3'].dropna(), bins=15, edgecolor='black', color='mediumseagreen')
plt.title('MMSE Score Distribution at Wave 3', fontsize=16)
plt.xlabel('MMSE Score', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('mmse_w3_distribution.png', dpi=300)
plt.close()

# 13. Plot Race and Ethnicity Distribution
# Map 'Race' codes to labels
race_mapping = {str(k): v for k, v in participants_meta['Race']['Levels'].items()}
participants_df['Race'] = map_categorical(participants_df, 'Race', race_mapping)

# Debug: Verify 'Race' mapping
print("\nRace Counts:")
print(participants_df['Race'].value_counts())

# Handle unmapped values if any
participants_df['Race'].fillna('Unknown', inplace=True)

# Plot Race distribution
race_counts = participants_df['Race'].value_counts()
plt.figure(figsize=(10, 6))
race_counts.plot(kind='bar', color='skyblue')
plt.title('Race Distribution of Participants', fontsize=16)
plt.xlabel('Race', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', alpha=0.75)
plt.savefig('race_distribution.png', dpi=300)
plt.close()

# Map 'Ethnicity' codes to labels
ethnicity_mapping = {str(k): v for k, v in participants_meta['Ethnicity']['Levels'].items()}
participants_df['Ethnicity'] = map_categorical(participants_df, 'Ethnicity', ethnicity_mapping)

# Debug: Verify 'Ethnicity' mapping
print("\nEthnicity Counts:")
print(participants_df['Ethnicity'].value_counts())

# Handle unmapped values if any
participants_df['Ethnicity'].fillna('Unknown', inplace=True)

# Plot Ethnicity distribution
ethnicity_counts = participants_df['Ethnicity'].value_counts()
plt.figure(figsize=(6, 6))
ethnicity_counts.plot(kind='bar', color='lightgreen')
plt.title('Ethnicity Distribution of Participants', fontsize=16)
plt.xlabel('Ethnicity', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.xticks(rotation=0)
plt.tight_layout()
plt.grid(axis='y', alpha=0.75)
plt.savefig('ethnicity_distribution.png', dpi=300)
plt.close()

# 14. Plot Handedness Score Distribution
plt.figure(figsize=(8, 6))
plt.hist(participants_df['HandednessScore'].dropna(), bins=5, edgecolor='black', color='orchid')
plt.title('Handedness Score Distribution', fontsize=16)
plt.xlabel('Handedness Score', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.xticks([0, 1, 2, 3, 4], ['Always Left', 'Usually Left', 'No Preference', 'Usually Right', 'Always Right'])
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('handedness_distribution.png', dpi=300)
plt.close()

# 15. Plot Education Level Distribution
# Map 'EduComp' codes to labels
edu_mapping = {str(k): v for k, v in participants_meta['EduComp']['Levels'].items()}
participants_df['EduComp'] = map_categorical(participants_df, 'EduComp', edu_mapping)

# Debug: Verify 'EduComp' mapping
print("\nEducation Level Counts:")
print(participants_df['EduComp'].value_counts())

# Handle unmapped values if any
participants_df['EduComp'].fillna('Unknown', inplace=True)

# Plot Education Level distribution
edu_counts = participants_df['EduComp'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
edu_counts.plot(kind='bar', color='goldenrod')
plt.title('Education Level of Participants', fontsize=16)
plt.xlabel('Highest Level of Education Completed', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', alpha=0.75)
plt.savefig('education_level_distribution.png', dpi=300)
plt.close()

# 16. Plot Time Intervals Between MRI Scans
# Time Interval between Wave 1 and Wave 2
plt.figure(figsize=(10, 6))
plt.hist(participants_df['MRIW1toW2'].dropna(), bins=15, edgecolor='black', color='teal')
plt.title('Time Interval Between Wave 1 and Wave 2 MRI Scans', fontsize=16)
plt.xlabel('Time Interval (years)', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('mri_time_interval_w1_w2.png', dpi=300)
plt.close()

# Time Interval between Wave 2 and Wave 3
plt.figure(figsize=(10, 6))
plt.hist(participants_df['MRIW2toW3'].dropna(), bins=15, edgecolor='black', color='slateblue')
plt.title('Time Interval Between Wave 2 and Wave 3 MRI Scans', fontsize=16)
plt.xlabel('Time Interval (years)', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('mri_time_interval_w2_w3.png', dpi=300)
plt.close()

# 17. Scatter Plot of Age vs. BMI at Wave 1
plt.figure(figsize=(10, 6))
plt.scatter(participants_df['AgeMRI_W1'], participants_df['BMI_W1'], alpha=0.7, color='darkorange')
plt.title('Age vs. BMI at Wave 1', fontsize=16)
plt.xlabel('Age (years)', fontsize=14)
plt.ylabel('BMI', fontsize=14)
plt.grid(True)
plt.savefig('age_vs_bmi_w1.png', dpi=300)
plt.close()

# 18. Histogram of Age Differences between Wave 1 and Wave 2
plt.figure(figsize=(10, 6))
plt.hist(participants_df['AgeDifference_W1_W2'].dropna(), bins=15, edgecolor='black', color='navy')
plt.title('Age Difference Between Wave 1 and Wave 2 MRI Scans', fontsize=16)
plt.xlabel('Age Difference (years)', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.grid(axis='y', alpha=0.75)
plt.savefig('age_difference_w1_w2.png', dpi=300)
plt.close()

# 19. Plot Number of Participants with MRI Data at Each Wave
waves = ['AgeMRI_W1', 'AgeMRI_W2', 'AgeMRI_W3']
participants_per_wave = participants_df[waves].notnull().sum()

plt.figure(figsize=(8, 6))
participants_per_wave.plot(kind='bar', color='mediumvioletred')
plt.title('Number of Participants with MRI Data at Each Wave', fontsize=16)
plt.xlabel('Wave', fontsize=14)
plt.ylabel('Number of Participants', fontsize=14)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()
plt.savefig('participants_per_wave.png', dpi=300)
plt.close()

# 20. Correlation Heatmap
plt.figure(figsize=(12, 10))
# Select numeric columns for correlation
numeric_cols = participants_df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = participants_df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Participants Data', fontsize=16)
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()

# 21. Save the updated processed data with additional metrics
participants_df.to_csv('participants_processed.tsv', sep='\t', index=False)

