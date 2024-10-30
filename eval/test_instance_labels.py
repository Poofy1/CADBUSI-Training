import pandas as pd

# Read the CSVs
poor_performing = pd.read_csv("F:/CODE/CADBUSI/CADBUSI-Training/eval/results/RMIL_OOD/poor_performing_instances_train.csv")
train_data = pd.read_csv("D:/DATA/CASBUSI/exports/export_oneLesions/TrainData.csv")

# Extract the accession number from the ID
poor_performing['accession_number'] = poor_performing['id'].apply(lambda x: int(x.split('_')[0]))

# Filter for positive targets (target = 1)
positive_cases = poor_performing[poor_performing['targets'] == 1]

# Merge with train data using accession number
merged_data = positive_cases.merge(train_data, left_on='accession_number', right_on='Accession_Number')

# Calculate statistics
total_positive_cases = len(positive_cases)
correct_malignant_labels = len(merged_data[merged_data['Has_Malignant'] == True])

ratio = correct_malignant_labels / total_positive_cases if total_positive_cases > 0 else 0

print(f"Total positive cases: {total_positive_cases}")
print(f"Cases with correct malignant label: {correct_malignant_labels}")
print(f"Ratio of correct labels: {ratio:.2f}")

# Optional: Display detailed mismatches
mismatches = merged_data[merged_data['Has_Malignant'] == False]
if len(mismatches) > 0:
    print("\nMismatched cases (where target=1 but Has_Malignant=False):")
    print(mismatches[['id', 'predictions', 'Has_Malignant', 'Has_Benign']])
    
# Extract all positive cases accession numbers from poor_performing
positive_cases_accession_numbers = set(positive_cases['accession_number'])

# Extract all accession numbers from train_data
train_data_accession_numbers = set(train_data['Accession_Number'])

# Find accession numbers that are in positive_cases but not in train_data
missing_accession_numbers = positive_cases_accession_numbers - train_data_accession_numbers

print(f"\nNumber of missing accession numbers: {len(missing_accession_numbers)}")
print("\nMissing accession numbers:")
print(sorted(list(missing_accession_numbers)))

# Let's also look at some sample rows from positive_cases that aren't getting matched
unmatched_cases = positive_cases[~positive_cases['accession_number'].isin(train_data_accession_numbers)]
print("\nSample of unmatched positive cases:")
print(unmatched_cases)

# Print the unique values in the original 'id' column for these unmatched cases
print("\nUnique original IDs in unmatched cases:")
print(unmatched_cases['id'].unique())