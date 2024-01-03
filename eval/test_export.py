import pandas as pd
import ast, os
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(current_dir)


def convert_string_to_list(string):
    try:
        return ast.literal_eval(string)
    except:
        return []
    
    
def get_birads_distribution(df, export_name):
    # Count the occurrences of each BI-RADS category
    bi_rads_counts = df['BI-RADS'].value_counts()
    ordered_categories = ['0', '1', '2', '3', '4', '4A', '4B', '4C', '5', '6']
    bi_rads_ordered = bi_rads_counts.reindex(ordered_categories).fillna(0)
    plt.figure(figsize=(10, 6))
    bars = bi_rads_ordered.plot(kind='bar', color='skyblue')
    plt.title(f'BI-RADS Distribution ({export_name})')
    plt.xlabel('BI-RADS')
    plt.ylabel('Accessions')
    plt.xticks(rotation=0)  # Rotate the x-axis labels to be horizontal

    # Annotate each bar with its count
    for bar in bars.patches:
        plt.annotate(format(bar.get_height(), '.0f'), 
                    (bar.get_x() + bar.get_width() / 2, 
                    bar.get_height()), ha='center', va='center',
                    size=10, xytext=(0, 8),
                    textcoords='offset points')

    # Save the plot
    full_path = f"{env}/results/BI-RADS_Distribution_{export_name}.png"
    plt.savefig(full_path, format='png')


def get_biopsy_distribution(malignant_df, benign_df, export_name):
    # Count the occurrences of each BI-RADS category
    bi_rads_malignant_counts = malignant_df['BI-RADS'].value_counts()
    bi_rads_benign_counts = benign_df['BI-RADS'].value_counts()
    ordered_categories = ['0', '1', '2', '3', '4', '4A', '4B', '4C', '5', '6']
    bi_rads_malignant_counts = bi_rads_malignant_counts.reindex(ordered_categories).fillna(0)
    bi_rads_benign_counts = bi_rads_benign_counts.reindex(ordered_categories).fillna(0)
    
    # Calculate percentages
    total_counts = bi_rads_malignant_counts + bi_rads_benign_counts
    malignant_percentages = (bi_rads_malignant_counts / total_counts * 100).fillna(0)
    benign_percentages = (bi_rads_benign_counts / total_counts * 100).fillna(0)

    # Plotting
    plt.figure(figsize=(10, 6))
    # Using softer color tones
    bars1 = plt.bar(ordered_categories, malignant_percentages, color='#ff9999', label='Malignant')  # Soft red
    bars2 = plt.bar(ordered_categories, benign_percentages, bottom=malignant_percentages, color='#9999ff', label='Benign')  # Soft blue

    plt.title(f'Biopsy Distribution by BI-RADS Category ({export_name})')
    plt.xlabel('BI-RADS')
    plt.ylabel('Percentage')
    plt.xticks(rotation=0)
    plt.legend()

    # Annotate each bar with its percentage
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only display if there is a notable height
                plt.annotate(f'{height:.1f}%', 
                            (bar.get_x() + bar.get_width() / 2, 
                            bar.get_y() + height/2), ha='center', va='center',
                            size=10, xytext=(0, 8),
                            textcoords='offset points')

    # Save the plot
    full_path = f"{env}/results/Biopsy_Distribution_{export_name}.png"
    plt.savefig(full_path, format='png')


def plot_age_vs_malignant_biopsies(df, export_name):

    # Group by age and count the number of malignant biopsies
    age_malignant_counts = df.groupby('Age').size()

    # Plotting
    plt.figure(figsize=(12, 6))
    age_malignant_counts.plot(kind='line', color='red')
    
    plt.title(f'Age vs. Malignant Biopsies ({export_name})')
    plt.xlabel('Age')
    plt.ylabel('Number of Malignant Biopsies')
    plt.grid(True)
    
    # Save the plot
    full_path = f"{env}/results/Age_vs_Malignant_Biopsies_{export_name}.png"
    plt.savefig(full_path, format='png')


def plot_size_vs_malignant_biopsies(df, export_name):
    df = df.copy()

    # Convert 'PatientSize' to numeric and remove invalid values
    df['PatientSize'] = pd.to_numeric(df['PatientSize'], errors='coerce')
    df = df.dropna(subset=['PatientSize'])
    df = df[df['PatientSize'] != 0]

    # Group by 'PatientSize' and count the number of malignant biopsies
    size_malignant_counts = df.groupby('PatientSize').size()

    # Apply Gaussian smoothing
    sigma = 2  # Standard deviation for Gaussian kernel
    smoothed_counts = gaussian_filter1d(size_malignant_counts, sigma)

    # Create an index for plotting (since Gaussian filtering returns an array)
    index = size_malignant_counts.index

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(index, smoothed_counts, color='blue')
    
    plt.title(f'Patient Size vs. Malignant Biopsies (Gaussian Smoothed) ({export_name})')
    plt.xlabel('Patient Size')
    plt.ylabel('Frequency of Malignant Biopsies')
    plt.grid(True)
    
    # Save the plot
    full_path = f"{env}/results/PatientSize_vs_Malignant_Biopsies_{export_name}.png"
    plt.savefig(full_path, format='png')
    


#export_name = 'export_11_11_2023'
export_name = 'export_12_26_2023'
export_location = f'D:/DATA/CASBUSI/exports/{export_name}/'
os.makedirs(f'{env}/results/', exist_ok=True)
image_df = pd.read_csv(f'{export_location}/ImageData.csv')
case_df = pd.read_csv(f'{export_location}/CaseStudyData.csv')
breast_df = pd.read_csv(f'{export_location}/BreastData.csv')
train_df = pd.read_csv(f'{export_location}/TrainData.csv')


# Filter data
image_df['Accession_Number'] = image_df['Accession_Number'].astype(int)
case_df['Accession_Number'] = case_df['Accession_Number'].astype(int)
breast_df['Accession_Number'] = breast_df['Accession_Number'].astype(int)

case_df = case_df.drop_duplicates(subset='Accession_Number')
case_df = case_df[case_df['Accession_Number'].isin(set(image_df['Accession_Number']))]
case_df = case_df[~case_df['Biopsy'].str.contains('unknown', na=False)]

breast_df = breast_df[breast_df['Accession_Number'].isin(set(image_df['Accession_Number']))]
breast_df = breast_df[breast_df['LesionCount'].astype(int) != 0]
breast_df = breast_df[breast_df['Has_Unknown'] == False]

# Filter for malignant cases
case_malignant_df = case_df[case_df['Biopsy'].str.contains('malignant', na=False)]

# Filter for cases that contain 'benign' but not 'malignant'
case_benign_df = case_df[case_df['Biopsy'].str.contains('benign', na=False) & ~case_df['Biopsy'].str.contains('malignant', na=False)]



print(f"Trainable Data Statistics for {export_name}:")

total_images = len(image_df)
print(f"Number of Images: {total_images}")

total_unique_patient_ids = case_df['Patient_ID'].nunique()
print(f"Number of Patient IDs: {total_unique_patient_ids}")

total_accessions = len(case_df)
print(f"Number of Accessions: {total_accessions}")

# Calculate revisit rate
visit_counts = case_df['Patient_ID'].value_counts()
average_visits_per_patient = visit_counts.mean()
print(f"Mean Visits per Patient: {average_visits_per_patient:.2f}")

# Get avergae number of images per accession
train_df['Images'] = train_df['Images'].apply(convert_string_to_list)
train_df['Images_Length'] = train_df['Images'].apply(len)
average_length = train_df['Images_Length'].mean()
print(f"Mean Number of 'Good' Images per Accession: {average_length:.2f}")

total_breasts = len(breast_df)
print(f"Number of Biopsied Breasts: {total_breasts}")

total_breasts = len(breast_df[(breast_df['Has_Malignant'] == False) & (breast_df['Has_Benign'] == True)])
print(f"Number of Benign Breasts: {total_breasts}")

total_breasts = len(breast_df[(breast_df['Has_Malignant'] == True) & (breast_df['Has_Benign'] == False)])
print(f"Number of Malignant Breasts: {total_breasts}")

total_breasts = len(breast_df[(breast_df['Has_Malignant'] == True) & (breast_df['Has_Benign'] == True)])
print(f"Number of Benign AND Malignant Breasts: {total_breasts}")

# Count the occurrences of each category in 'Study_Laterality'
laterality_counts = case_df['Study_Laterality'].value_counts()
total_cases = laterality_counts.sum()
laterality_percentage = (laterality_counts / total_cases) * 100
print("\nExam Laterality Distribution:")
print(f"RIGHT: {laterality_percentage.get('RIGHT', 0):.2f}%")
print(f"LEFT: {laterality_percentage.get('LEFT', 0):.2f}%")
print(f"BILATERAL: {laterality_percentage.get('BILATERAL', 0):.2f}%")


get_birads_distribution(case_df, export_name)

get_biopsy_distribution(case_malignant_df, case_benign_df, export_name)

plot_age_vs_malignant_biopsies(case_malignant_df, export_name)

plot_size_vs_malignant_biopsies(case_malignant_df, export_name)

print("\nFinished drawing graphs to results folder")