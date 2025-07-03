from fastai.vision.all import *


def create_selection_mask(train_bag_logits, include_ratio, positive_bags_only=True):
    """
    Creates a selection mask for bag instances based on confidence scores.

    Parameters:
    train_bag_logits (dict): A dictionary where keys are bag IDs and values are dictionaries
                             containing 'instance_predictions' and 'bag_label'. 
                             'instance_predictions' should be a numpy array of probabilities 
                             for each instance in the bag. Each probability should be a float 
                             between 0 and 1, where values closer to 0 or 1 indicate higher confidence.
                             Example: {1: {'instance_predictions': [0.1, 0.9, 0.6], 'bag_label': 1}, 
                                      2: {'instance_predictions': [0.3, 0.7, 0.4], 'bag_label': 0}}

    include_ratio (float): A value between 0 and 1 indicating the proportion of instances
                           to include in the selection. For example, 0.5 means include
                           the top 50% most confident predictions.
                           
    positive_bags_only (bool): If True, only selects the most confident positive 
                                          instance (highest probability > 0.5) from each positive bag.
                                          Ignores include_ratio when True.

    Returns:
    dict: Dictionary where values contain:
          1. Pseudo-label array: 1 (positive), 0 (negative), -1 (unlabeled/not selected)
          2. Original probabilities

    Note: The function assumes that probabilities closer to 0 or 1 indicate higher confidence,
          while probabilities closer to 0.5 indicate lower confidence.
    """
    
    # Initialize combined_dict with all -1 for masks (not selected by default) and predictions
    combined_dict = {}
    for bag_id, data in train_bag_logits.items():
        bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
        probs = data['instance_predictions']  # Extract instance predictions
        mask = np.full(len(probs), -1, dtype=int)  # Initialize mask to -1 (not selected)
        pred_list = []
        
        # Store all predictions
        for prob in probs:
            prob_value = prob.item() if hasattr(prob, 'item') else float(prob)
            pred_list.append(prob_value)
        
        combined_dict[bag_id_int] = [mask, pred_list]
    
    combined_probs = []
    original_indices = []
    predictions = []
    
    # Loop through train_bag_logits to process probabilities
    for bag_id, data in train_bag_logits.items():
        bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
        probs = combined_dict[bag_id_int][1]  # Use the already processed predictions
        
        for i, prob in enumerate(probs):
            combined_probs.append(prob)
            original_indices.append((bag_id_int, i))
            predictions.append(prob)
                
    if positive_bags_only:
        selected_count = 0
        
        # For each positive bag, select the most confident instance + top include_ratio instances
        for bag_id, data in train_bag_logits.items():
            bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
            bag_label = data['bag_label']
            
            # Convert bag_label to scalar if it's a tensor/array
            if hasattr(bag_label, 'item'):
                bag_label = bag_label.item()
            elif isinstance(bag_label, np.ndarray):
                bag_label = bag_label.item() if bag_label.size == 1 else bag_label[0]
            
            # Only process positive bags
            if bag_label == 1:
                probs = combined_dict[bag_id_int][1]  # Get the prediction list
                
                # Always select the instance with the highest probability (guaranteed)
                best_idx = np.argmax(probs)
                combined_dict[bag_id_int][0][best_idx] = 1  # Set mask to 1 (positive)
                
                # Also select top include_ratio instances from this bag (only those above 0.5)
                # Filter instances that are above 0.5
                above_threshold_indices = [i for i, prob in enumerate(probs) if prob > 0.5]
                
                if len(above_threshold_indices) > 0:
                    # Calculate how many additional instances to select from those above 0.5
                    num_additional = int(len(above_threshold_indices) * include_ratio)
                    
                    if num_additional > 0:
                        # Get probabilities for above-threshold instances and sort by probability
                        above_threshold_probs = [(i, probs[i]) for i in above_threshold_indices]
                        above_threshold_probs.sort(key=lambda x: x[1], reverse=True)  # Sort by prob, highest first
                        
                        # Select top num_additional instances
                        top_indices = [idx for idx, _ in above_threshold_probs[:num_additional]]
                        
                        # Mark all top instances as positive
                        for idx in top_indices:
                            combined_dict[bag_id_int][0][idx] = 1
                
                # Count unique selected instances in this bag
                selected_in_bag = np.sum(combined_dict[bag_id_int][0] == 1)
                selected_count += selected_in_bag
        
        print(f'Selected {selected_count} instances from positive bags (guaranteed best + top {include_ratio:.1%} of instances >0.5 per bag)')
            
    else:
        total_predictions = len(combined_probs)
        predictions_included = int(total_predictions * include_ratio)
        print(f'Including Predictions: {include_ratio:.2f} ({predictions_included})')

        # Rank instances based on their confidence (distance from 0.5)
        confidence_scores = np.abs(np.array(combined_probs) - 0.5)
        top_indices = np.argsort(-confidence_scores)[:predictions_included]

        # Set mask based on selection
        for idx in top_indices:
            original_bag_id, original_position = original_indices[idx]
            prob = combined_probs[idx]
            # Update mask based on probability: 0 if below 0.5, 1 if above 0.5
            combined_dict[original_bag_id][0][original_position] = int(prob > 0.5)

    return combined_dict





def create_momentum_predictions(train_bag_logits, previous_predictions=None, momentum=0.9):
    """
    Creates float predictions for all instances using momentum-based updates, starting from 0.5.
    All predictions are clamped between 0 and 1.

    Parameters:
    train_bag_logits (dict): Dictionary with bag IDs as keys and lists/tensors of probabilities
                            as values.
    previous_predictions (dict): Optional. Previous predictions dictionary with the same structure
                               as the output. If None, initializes all predictions to 0.5.
    momentum (float): Momentum factor between 0 and 1. Higher values give more weight to
                     previous predictions. Default is 0.9.

    Returns:
    dict: A dictionary where keys are bag IDs and values are lists containing:
          1. A numpy array of float predictions (clamped between 0 and 1)
          2. A list of the current input probabilities
    """
    combined_dict = {}
    
    for bag_id, probs in train_bag_logits.items():
        bag_id_int = bag_id.item() if isinstance(bag_id, torch.Tensor) else bag_id
        
        # Convert current probabilities to list of floats
        current_probs = [p.item() if hasattr(p, 'item') else p for p in probs]
        
        # Initialize or get previous predictions
        if previous_predictions is None or bag_id_int not in previous_predictions:
            # Start with 0.5 for all instances if no previous predictions
            prev_predictions = [0.5] * len(probs)
        else:
            prev_predictions = previous_predictions[bag_id_int][0]
        
        # Apply momentum update and clamp values between 0 and 1
        updated_predictions = [
            np.clip(momentum * prev_pred + (1 - momentum) * curr_prob, 0, 1)
            for prev_pred, curr_prob in zip(prev_predictions, current_probs)
        ]
        
        # Store results as numpy array for predictions
        combined_dict[bag_id_int] = [
            np.array(updated_predictions),  # Float predictions clamped to [0,1]
            current_probs                   # Current input probabilities
        ]
    
    return combined_dict