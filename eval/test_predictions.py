import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_predictions(csv_path):
    # Read the CSV file
    data = pd.read_csv(csv_path)
    
    # 1. Analyze label distribution
    label_dist = data['targets'].value_counts()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot label distribution
    label_dist.plot(kind='bar', ax=ax1)
    ax1.set_title('Label Distribution')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Count')
    
    # Plot prediction distributions by label
    for label in data['targets'].unique():
        predictions = data[data['targets'] == label]['predictions']
        sns.kdeplot(data=predictions, label=f'Label {label}', ax=ax2)
    
    ax2.set_title('Prediction Distribution by Label')
    ax2.set_xlabel('Prediction Value')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nLabel Distribution:")
    print(label_dist)
    print("\nPrediction Statistics by Label:")
    for label in data['targets'].unique():
        predictions = data[data['targets'] == label]['predictions']
        print(f"\nLabel {label}:")
        print(f"Mean: {predictions.mean():.4f}")
        print(f"Std: {predictions.std():.4f}")
        print(f"Min: {predictions.min():.4f}")
        print(f"Max: {predictions.max():.4f}")

    # Find poor performing instances
    poor_performing = data[
        ((data['targets'] == 0) & (data['predictions'] >= 0.9)) |
        ((data['targets'] == 1) & (data['predictions'] <= 0.1))
    ]

    # Get output path in same folder
    output_path = os.path.join(os.path.dirname(csv_path), 'poor_performing_instances_train.csv')
    
    # Save poor performing instances
    poor_performing.to_csv(output_path, index=False)
    print(f"\nPoor performing instances saved to: {output_path}")
    print(f"Number of poor performing instances: {len(poor_performing)}")

# Use your file path
csv_path = "F:/CODE/CADBUSI/CADBUSI-Training/eval/results/RMIL_OOD/instance_predictions_train.csv"
analyze_predictions(csv_path)