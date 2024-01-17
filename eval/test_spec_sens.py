import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming current_dir and parent_dir have been defined as shown in the initial code
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Read the data
df_failed_cases = pd.read_csv(f"{parent_dir}/results/failed_cases.csv")

# Sort the DataFrame by the 'Prediction' column
df_sorted = df_failed_cases.sort_values(by='Prediction')

# Initialize lists to store sensitivity, specificity, and accuracy values
sensitivity = []
specificity = []
accuracy = []

# Variables to store the best threshold and its accuracy
best_threshold = None
best_accuracy = 0

# Calculate sensitivity, specificity, and accuracy for each threshold
for threshold in df_sorted['Prediction'].unique():
    # True positives and negatives, false positives and negatives
    TP = len(df_sorted[(df_sorted['True_Label'] == 1) & (df_sorted['Prediction'] >= threshold)])
    TN = len(df_sorted[(df_sorted['True_Label'] == 0) & (df_sorted['Prediction'] < threshold)])
    FP = len(df_sorted[(df_sorted['True_Label'] == 0) & (df_sorted['Prediction'] >= threshold)])
    FN = len(df_sorted[(df_sorted['True_Label'] == 1) & (df_sorted['Prediction'] < threshold)])
    
    # Calculate sensitivity, specificity, and accuracy
    sens = TP / (TP + FN) if (TP + FN) != 0 else 0
    spec = TN / (TN + FP) if (TN + FP) != 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN)

    sensitivity.append(sens)
    specificity.append(spec)
    accuracy.append(acc)

    # Update the best threshold and accuracy
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold

# Print the best threshold and its accuracy
print(f"Best Threshold: {best_threshold}")
print(f"Accuracy at best threshold: {best_accuracy}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['Prediction'].unique(), sensitivity, label='Sensitivity')
plt.plot(df_sorted['Prediction'].unique(), specificity, label='Specificity')
plt.plot(df_sorted['Prediction'].unique(), accuracy, label='Accuracy')
plt.xlabel('Prediction Threshold')
plt.ylabel('Metric Value')
plt.title('Sensitivity, Specificity, and Accuracy by Prediction Threshold')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(f"{parent_dir}/results/sensitivity_specificity_accuracy_graph.png")

# Optionally, close the plot if not showing it
plt.close()
