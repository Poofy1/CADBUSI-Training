from fastai.vision.all import *


def create_selection_mask(train_bag_logits, include_ratio):
    
    """
    Creates a selection mask for bag instances based on confidence scores.

    Parameters:
    train_bag_logits (dict): A dictionary where keys are bag IDs and values are lists or tensors
                             of probabilities for each instance in the bag. Each probability
                             should be a float between 0 and 1, where values closer to 0 or 1
                             indicate higher confidence.
                             Example: {1: [0.1, 0.9, 0.6], 2: [0.3, 0.7, 0.4]}

    include_ratio (float): A value between 0 and 1 indicating the proportion of instances
                           to include in the selection. For example, 0.5 means include
                           the top 50% most confident predictions.

    Returns:
    dict: A dictionary where keys are bag IDs and values are lists containing two elements:
          1. A numpy array mask where 1 indicates selected instances, 0 indicates
             unselected instances, and -1 indicates instances not considered for selection.
          2. A list of the original probabilities for each instance.

    Note: The function assumes that probabilities closer to 0 or 1 indicate higher confidence,
          while probabilities closer to 0.5 indicate lower confidence.
    """
    
    combined_probs = []
    original_indices = []
    predictions = []
    
    # Loop through train_bag_logits to process probabilities
    for bag_id, probs in train_bag_logits.items():
        # Convert tensor bag_id to integer if necessary
        bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
        for i, prob in enumerate(probs):
            combined_probs.append(prob.item())
            original_indices.append((bag_id_int, i))
            predictions.append(prob.item())

    total_predictions = len(combined_probs)
    predictions_included = int(total_predictions * include_ratio)
    print(f'Including Predictions: {include_ratio:.2f} ({predictions_included})')

    # Rank instances based on their confidence (distance from 0.5)
    confidence_scores = np.abs(np.array(combined_probs) - 0.5)
    top_indices = np.argsort(-confidence_scores)[:predictions_included]

    # Initialize combined_dict with all -1 for masks (not selected by default) and placeholders for predictions
    combined_dict = {}
    for bag_id, probs in train_bag_logits.items():
        bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
        mask = np.full(len(probs), -1, dtype=int)  # Initialize mask to -1 (not selected)
        pred_list = [None] * len(probs)  # Initialize prediction list
        combined_dict[bag_id_int] = [mask, pred_list]

    # Update predictions in combined_dict for all instances
    for idx, (bag_id_int, pos) in enumerate(original_indices):
        combined_dict[bag_id_int][1][pos] = predictions[idx]  # Update prediction

    # Set mask based on selection
    for idx in top_indices:
        original_bag_id, original_position = original_indices[idx]
        prob = combined_probs[idx]
        # Update mask based on probability: 0 if below 0.5, 1 if above 0.5
        combined_dict[original_bag_id][0][original_position] = int(prob > 0.5)

    return combined_dict
