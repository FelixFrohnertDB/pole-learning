import numpy as np
import gc
import pickle
import glob
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostClassifier
import sys

raw = int(sys.argv[1])
print(f"Raw mode: {raw}")

_feature_names = glob.glob('feature_names*.npy')

# Extract the part between the last underscore and .npy
identifier = _feature_names[0].split('_')[-1].replace('.npy', '')
feature_names = np.load(_feature_names[0], allow_pickle=True)

if raw:
    print("Processing raw features...")
    # Initialize an empty list to hold all feature arrays
    feature_list = []

    # Loop from 0 to 34 (inclusive)
    for i in range(35):
        # Format the filename with leading zeros (e.g., P00, P01, ..., P34)
        filename = f"rawFeatures/P{i:02d}_intensity.pkl"
        
        # Load the file and append to the list
        features = np.array(np.load(filename, allow_pickle=True))
        feature_list.append(features)

    # Concatenate all arrays along axis 0
    X_arr = np.concatenate(feature_list, axis=0)

else:
   
    input_files = glob.glob('features_filtered*.npy')
    if not input_files:
        raise FileNotFoundError("No features_filtered*.npy file found in current directory")
    input_path = input_files[0]

    # Load memory-mapped array
    print(f"Loading features from {input_path}")
    X_arr = np.load(input_path, mmap_mode='r')

    # Apply variance threshold more efficiently
    print(f"Original feature shape: {X_arr.shape}")
    sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
    X_arr = sel.fit_transform(X_arr)
    print(f"Shape after variance threshold: {X_arr.shape}")

    # Get the corresponding feature names
    selected_features_mask = sel.get_support()
    feature_names = feature_names[selected_features_mask]

    
# Class mapping dictionary
class_to_poles = {
    0: [0, 0, 0],  # 1 pole on [bt]
    1: [1, 0, 0],  # 1 pole on [bb]
    2: [0, 1, 0],  # 1 pole on [tb]
    3: [0, 0, 1],  # 1 pole on [bt] and 1 pole on [bb]
    4: [2, 0, 0],  # 1 pole on [bb] and 1 pole on [tb]
    5: [0, 2, 0],  # 1 pole on each of [bt], [bb], and [tb]
    6: [0, 0, 2],  # 2 poles on [bb] and 1 pole on [tb]
    7: [1, 1, 0],  # 1 pole on [bb] and 2 poles on [tb]
    8: [1, 0, 1],
    9: [0, 1, 1],
    10: [3, 0, 0],
    11: [0, 3, 0],
    12: [0, 0, 3],
    13: [2, 1, 0],
    14: [2, 0, 1],
    15: [1, 2, 0],
    16: [0, 2, 1],
    17: [1, 0, 2],
    18: [0, 1, 2],
    19: [1, 1, 1],
    20: [4, 0, 0],
    21: [0, 4, 0],
    22: [0, 0, 4],
    23: [3, 1, 0],
    24: [3, 0, 1],
    25: [1, 3, 0],
    26: [0, 3, 1],
    27: [1, 0, 3],
    28: [0, 1, 3],
    29: [2, 2, 0],
    30: [2, 0, 2],
    31: [0, 2, 2],
    32: [2, 1, 1],
    33: [1, 2, 1],
    34: [1, 1, 2],
}

def reconvert_labels(new_labels, class_to_poles):
    """
    Convert a label array back to the original class labels.

    Parameters:
        new_labels (np.ndarray): A 2D numpy array with shape (n, 3), where each row is a pole representation.
        class_to_poles (dict): Mapping from class labels to pole representations.

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

def convert_labels(old_labels, class_to_poles):
    """
    Convert class labels to pole representation.
    
    Parameters:
        old_labels (np.ndarray): Array of class labels.
        class_to_poles (dict): Mapping from class labels to pole representations.
        
    Returns:
        np.ndarray: A 2D array with pole representations.
    """
    # Initialize a new array with shape (length of old labels, 3)
    new_labels = np.zeros((len(old_labels), 3), dtype=int)
    
    # Populate the new label array based on the mapping
    for i, label in enumerate(old_labels):
        new_labels[i] = class_to_poles[label]
    
    return new_labels

# File paths
file_prefix = "rawFeatures/P"
file_suffix = "_intensity.pkl"
num_files = 35

# Load labels
print("Loading labels...")
y_arr_classification = np.array([
    np.tile(i, np.load(f"{file_prefix}{0:02d}{file_suffix}", allow_pickle=True).shape[0]) 
    for i in np.arange(num_files)
]).flatten()

# Convert labels to regression format
# y_arr_regression = convert_labels(y_arr_classification, class_to_poles) * 1.0

# Train initial model to get feature importance
print("Training initial model for feature importance...")
rg = CatBoostClassifier(
    iterations=1000,
    verbose=50,
    early_stopping_rounds=10
)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X_arr, y_arr_classification, random_state=42, test_size=0.8
)

# Fit model
rg.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)
print("Initial model training complete")

# Step 2: Sort feature indices based on importance
print("Sorting features by importance...")
sorted_indices = np.argsort(rg.get_feature_importance())[::-1]


feature_importances_dict = {}

for n,v in zip(feature_names[sorted_indices],np.sort(rg.get_feature_importance())[::-1]):
    feature_importances_dict[n] = v

output_path = f'feature_importances_dict_{identifier}.pkl' if not raw else f'feature_raw_importances_dict_{identifier}.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(feature_importances_dict, f)


output_path = f"sorted_feature_names_{identifier}.npy" if not raw else f"sorted_feature_raw_names_{identifier}.npy"
np.save(output_path, feature_names[sorted_indices])

# Step 3: Set up dimensions and output memory-mapped file
n_samples, n_features = X_arr.shape
output_path = f'full_features_sorted_{n_samples}_{n_features}_{identifier}.npy' if not raw else f'full_features_sorted_raw_{n_samples}_{n_features}_{identifier}.npy'

# Create a new memory-mapped file for the sorted features
X_sorted = np.memmap(output_path, dtype='float32', mode='w+', shape=(n_samples, n_features))

# Step 4: Chunked reordering to avoid memory issues
print(f"Reordering features in chunks to {output_path}...")
chunk_size = 1000
for i in range(0, n_samples, chunk_size):
    end = min(i + chunk_size, n_samples)
    chunk = X_arr[i:end, :]  # doesn't load everything
    X_sorted[i:end, :] = chunk[:, sorted_indices]

# Step 5: Flush to disk and release memory
X_sorted.flush()
del X_arr
del X_sorted
# del rg
gc.collect()  # force cleanup

acc = accuracy_score(rg.predict(X_val),y_val)
print("Done: Acc", acc)

# # Step 6: Reload reordered array with memory-mapping
# print("Loading sorted features...")
# X_arr = np.memmap(output_path, dtype='float32', mode='r', shape=(n_samples, n_features))

# print("Starting cross-validation with different feature counts...")
# # Define feature counts to test using log-spaced percentages
# feature_percentages = np.logspace(-2, 0, 5)[-1]  # From 1% to 100% of features
# feature_counts = np.unique(
#     np.clip((feature_percentages * X_arr.shape[1]).astype(int), 1, X_arr.shape[1])
# )

# # Loop through different feature counts
# for d in feature_counts:
#     print(f"\nEvaluating with top {d} features ({d/X_arr.shape[1]*100:.2f}% of total)")
#     fold_accuracies = []

#     # Use k-fold cross-validation
#     kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     for fold, (train_index, val_index) in enumerate(kf.split(X_arr)):
#         print(f"  Fold {fold+1}/5")
#         # Extract current fold data with selected feature count
#         X_train, X_val = X_arr[train_index][:, :d], X_arr[val_index][:, :d]
#         y_train, y_val = y_arr_regression[train_index], y_arr_regression[val_index]

#         # Train model
#         model = CatBoostRegressor(
#             iterations=1000,
#             verbose=0,
#             loss_function="MultiRMSE",
#             early_stopping_rounds=10
#         )

#         model.fit(
#             X_train, y_train, 
#             eval_set=(X_val, y_val), 
#             use_best_model=True,
#             plot=False
#         )

#         # Save model
#         model_path = f"models/catboost_model_{d}_fold{fold}.cbm" if not raw else f"models/catboost_model_{d}_fold{fold}_raw.cbm"
#         model.save_model(model_path)
#         print(f"  Model saved to {model_path}")

#         # Predict and evaluate
#         y_pred = np.abs(np.round(model.predict(X_val)))
#         acc = accuracy_score(
#             reconvert_labels(y_pred, class_to_poles),
#             reconvert_labels(y_val, class_to_poles)
#         )
#         fold_accuracies.append(acc)
#         print(f"  Fold accuracy: {acc:.4f}")

#         # Clean up
#         del model, X_train, X_val, y_train, y_val
#         gc.collect()

#     # Save fold accuracies
#     acc_file = f"fold_accuracies_{d}.pkl" if not raw else f"fold_accuracies_{d}_raw.pkl"
#     with open(acc_file, "wb") as f:
#         pickle.dump({d: fold_accuracies}, f)
    

# print("Pipeline execution complete!")