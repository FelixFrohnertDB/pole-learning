import numpy as np 
from scipy.optimize import minimize 

def reconvert_labels(new_labels, class_to_poles):
    """
    Convert a label array back to the original class labels.

    Parameters:
        new_labels (np.ndarray): A 2D numpy array with shape (n, 3), where each row is a pole representation.

    Returns:
        np.ndarray: A 1D numpy array with the original class labels.
    """
    # Create a reverse mapping from pole lists to class labels
    poles_to_class = {tuple(v): k for k, v in class_to_poles.items()}

    # Initialize an array for the original class labels
    old_labels = np.zeros(len(new_labels), dtype=int)

    # Map each row of the input to its corresponding class label
    for i, pole in enumerate(new_labels):
        old_labels[i] = poles_to_class.get(tuple(pole), 0)  # Default to class 0 if tuple not found

    return old_labels

def check_duplicate_values(data_dict):
    values = list(data_dict.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if np.array_equal(values[i], values[j]):
                return True, (list(data_dict.keys())[i], list(data_dict.keys())[j])  # Return True and the keys
    return False, None

def convert_labels(old_labels, class_to_poles):
    # Initialize a new array with shape (length of old labels, 3)
    new_labels = np.zeros((len(old_labels), 3), dtype=int)
    
    # Populate the new label array based on the mapping
    for i, label in enumerate(old_labels):
        new_labels[i] = class_to_poles[label]
    
    return new_labels


def round_with_thresholds(preds, best_thresholds, class_to_poles):
    """
    Round predictions using class-specific thresholds based on the mapping of classes to poles.

    Parameters:
    - preds (numpy.ndarray): Array of predictions to be rounded (shape: [n_samples, 3]).
    - best_thresholds (list or dict): List or dict of thresholds for all 8 possible classes.
    - class_to_poles (dict): Mapping of classes (0-7) to their pole positions.

    Returns:
    - numpy.ndarray: Rounded predictions with the same shape as `preds`.
    """
    rounded_preds = np.empty_like(preds, dtype=int)

    # Map thresholds to the specific pole positions (columns in `preds`)
    pole_thresholds = np.zeros(3)  # Initialize thresholds for the 3 poles
    for class_label, poles in class_to_poles.items():
        for pole_idx, count in enumerate(poles):
            if count > 0:  # If the pole is relevant to this class
                pole_thresholds[pole_idx] = best_thresholds[class_label]

    # Apply thresholds to round predictions
    for i in range(preds.shape[1]):
        rounded_preds[:, i] = (preds[:, i] + pole_thresholds[i]).astype(int)

    return rounded_preds


def evaluate_accuracy(thresholds, preds, y_true, class_to_poles):
    """
    Evaluate the mean accuracy across all classes given thresholds.

    Parameters:
    - thresholds (np.ndarray): Thresholds for each class (length 8).
    - preds (np.ndarray): Array of predictions (n_samples, 3).
    - y_true (np.ndarray): Array of true labels (n_samples, 3).
    - class_to_poles (dict): Mapping of classes to pole configurations.

    Returns:
    - float: Negative mean accuracy (to minimize with `scipy.optimize`).
    """
   
    adjusted_preds = round_with_thresholds(preds, thresholds, class_to_poles)
    
    return -np.sum(np.sum(adjusted_preds == y_true, axis=1) == 3)/y_true.shape[0]

def optimize_thresholds(preds, y_true, class_to_poles):
    """
    Optimize thresholds to maximize mean accuracy.

    Parameters:
    - preds (np.ndarray): Array of predictions (n_samples, 3).
    - y_true (np.ndarray): Array of true labels (n_samples, 3).
    - class_to_poles (dict): Mapping of classes to pole configurations.

    Returns:
    - np.ndarray: Optimized thresholds for each class (length 8).
    """
    # Initial thresholds (e.g., 0.5 for all classes)
    initial_thresholds = np.full(len(class_to_poles), 0.5)

    # Bounds for thresholds (e.g., between 0 and 1)
    bounds = [(0, 1) for _ in range(len(class_to_poles))]

    # Optimize using scipy
    result = minimize(
        evaluate_accuracy,
        x0=initial_thresholds,
        args=(preds, y_true, class_to_poles),
        bounds=bounds,
        method='Nelder-Mead'
    )

    # Print the result of optimization
    print(f"Optimization result: {result}")
    return result.x