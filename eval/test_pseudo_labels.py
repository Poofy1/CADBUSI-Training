import pickle
import numpy as np
from collections import Counter

def analyze_selection_mask(file_path):
    # Load the selection mask
    with open(file_path, 'rb') as file:
        selection_mask = pickle.load(file)

    total_instances = 0
    selected_instances = 0
    label_counts = Counter()
    bag_sizes = []
    selected_per_bag = []

    for bag_id, (mask, predictions) in selection_mask.items():
        bag_size = len(mask)
        total_instances += bag_size
        bag_sizes.append(bag_size)

        selected_in_bag = np.sum(np.array(mask) != -1)
        selected_instances += selected_in_bag
        selected_per_bag.append(selected_in_bag)

        label_counts.update(mask)

    # Remove -1 from label counts as it represents unselected instances
    if -1 in label_counts:
        del label_counts[-1]

    print(f"Total bags: {len(selection_mask)}")
    print(f"Total instances: {total_instances}")
    print(f"Selected instances: {selected_instances} ({selected_instances/total_instances:.2%})")
    print(f"Average bag size: {np.mean(bag_sizes):.2f}")
    print(f"Average selected per bag: {np.mean(selected_per_bag):.2f}")
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  Label {label}: {count} ({count/selected_instances:.2%})")

    print("\nBag size distribution:")
    bag_size_counts = Counter(bag_sizes)
    for size, count in sorted(bag_size_counts.items()):
        print(f"  Size {size}: {count} bags")

if __name__ == "__main__":
    file_path = 'F:\CODE\CASBUSI\CASBUSI-Training\models\Head_Palm2_CASBUSI_224_2_efficientnet_b0\export_oneLesions_efficientnet_b0_1\selection_mask.pkl'
    analyze_selection_mask(file_path)