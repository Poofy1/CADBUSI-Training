import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

# Assuming current_dir and parent_dir have been defined as shown in the initial code
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Read the data
target_file = "export_oneLesions_predictions.csv"
df = pd.read_csv(f"{current_dir}/results/{target_file}")

# Filter only validation set results
df_val = df[df['Category'] == 'val']

# Sort the DataFrame by the 'Prediction' column
df_sorted = df_val.sort_values(by='Prediction', ascending=False)

# Set target specificity
target_specificity = 0.80

# Initialize lists to store sensitivity, specificity, and accuracy values
sensitivity = []
specificity = []
accuracy = []

# Variables to store the best threshold and its metrics
best_threshold = None
best_accuracy = 0
best_sensitivity = 0
best_specificity = 0
best_ppv = 0
best_npv = 0

# Calculate AUC
auc = roc_auc_score(df_sorted['Has_Malignant'], df_sorted['Prediction'])

# Calculate sensitivity, specificity, and accuracy for each threshold
thresholds = np.linspace(0, 1, 1000)  # Create 1000 evenly spaced thresholds for more precision
target_threshold = None
target_metrics = None

for threshold in thresholds:
    # True positives and negatives, false positives and negatives
    TP = len(df_sorted[(df_sorted['Has_Malignant'] == 1) & (df_sorted['Prediction'] >= threshold)])
    TN = len(df_sorted[(df_sorted['Has_Malignant'] == 0) & (df_sorted['Prediction'] < threshold)])
    FP = len(df_sorted[(df_sorted['Has_Malignant'] == 0) & (df_sorted['Prediction'] >= threshold)])
    FN = len(df_sorted[(df_sorted['Has_Malignant'] == 1) & (df_sorted['Prediction'] < threshold)])
    
    # Calculate sensitivity, specificity, and accuracy
    sens = TP / (TP + FN) if (TP + FN) != 0 else 0
    spec = TN / (TN + FP) if (TN + FP) != 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN)
    ppv = TP / (TP + FP) if (TP + FP) != 0 else 0
    npv = TN / (TN + FN) if (TN + FN) != 0 else 0

    sensitivity.append(sens)
    specificity.append(spec)
    accuracy.append(acc)

    # Find the threshold that achieves the target specificity
    if target_threshold is None and spec >= target_specificity:
        target_threshold = threshold
        target_metrics = (acc, sens, spec, ppv, npv)

    # Update the best threshold and metrics
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        best_sensitivity = sens
        best_specificity = spec
        best_ppv = ppv
        best_npv = npv

# Print the metrics for the best accuracy threshold
print("Model Performance Metrics (Validation Set) - Best Accuracy:")
print(f"* Accuracy: {best_accuracy:.2%}")
print(f"* AUC: {auc:.2%}")
print(f"* Sensitivity: {best_sensitivity:.2%}")
print(f"* Specificity: {best_specificity:.2%}")
print(f"* PPV: {best_ppv:.2%}")
print(f"* NPV: {best_npv:.2%}")
print(f"* Best Threshold: {best_threshold:.2f}")

# Print the metrics for the target specificity threshold
print(f"\nModel Performance Metrics (Validation Set) - Target Specificity ({target_specificity:.0%}):")
print(f"* Accuracy: {target_metrics[0]:.2%}")
print(f"* AUC: {auc:.2%}")
print(f"* Sensitivity: {target_metrics[1]:.2%}")
print(f"* Specificity: {target_metrics[2]:.2%}")
print(f"* PPV: {target_metrics[3]:.2%}")
print(f"* NPV: {target_metrics[4]:.2%}")
print(f"* Threshold: {target_threshold:.2f}")

# Plotting Performance Metrics
plt.figure(figsize=(8, 8))
plt.plot(thresholds, sensitivity, label='Sensitivity')
plt.plot(thresholds, specificity, label='Specificity')
plt.plot(thresholds, accuracy, label='Accuracy')
plt.axvline(x=best_threshold, color='r', linestyle='--', label='Best Accuracy Threshold')
plt.axvline(x=target_threshold, color='g', linestyle='--', label=f'Target Specificity Threshold ({target_specificity:.0%})')
plt.xlabel('Threshold')
plt.ylabel('Metric Value')
plt.title('Performance Metrics vs. Threshold')
plt.legend()
plt.grid(True)

# Save the performance metrics figure
plt.savefig(f"{current_dir}/results/Performance_metrics_graph.png")

# Plotting AUC graph
fpr, tpr, roc_thresholds = roc_curve(df_sorted['Has_Malignant'], df_sorted['Prediction'])

plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')

# Find the point on the ROC curve corresponding to the target specificity threshold
target_fpr = 1 - target_metrics[2]  # 1 - specificity
target_tpr = target_metrics[1]  # sensitivity

# Find the point on the ROC curve corresponding to the 0.5 threshold
threshold_05 = 0.5
fpr_05 = 1 - specificity[np.argmin(np.abs(thresholds - threshold_05))]
tpr_05 = sensitivity[np.argmin(np.abs(thresholds - threshold_05))]

# Plot both points
plt.plot(target_fpr, target_tpr, 'ro', markersize=10, label=f'Threshold at {target_specificity:.0%} Specificity')
plt.plot(fpr_05, tpr_05, 'go', markersize=10, label='Threshold at 0.5')

# Add crosshairs for both points
plt.axvline(x=target_fpr, color='red', linestyle=':', alpha=0.8)
plt.axhline(y=target_tpr, color='red', linestyle=':', alpha=0.8)
plt.axvline(x=fpr_05, color='green', linestyle=':', alpha=0.8)
plt.axhline(y=tpr_05, color='green', linestyle=':', alpha=0.8)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)

# Annotate both points
plt.annotate(f'Threshold: {target_threshold:.2f}',
             xy=(target_fpr, target_tpr), xycoords='data',
             xytext=(10, -10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.annotate(f'Threshold: 0.5',
             xy=(fpr_05, tpr_05), xycoords='data',
             xytext=(10, 10), textcoords='offset points',
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# Save the AUC figure
plt.savefig(f"{current_dir}/results/AUC_graph.png")