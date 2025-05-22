import numpy as np
import gc
import pickle
import glob
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostRegressor, Pool
import sys

raw = int(sys.argv[1])
print(f"Raw mode: {raw}")

# Automatically find the matching file
input_files = glob.glob('features_selected*.npy')
if not input_files:
    raise FileNotFoundError("No features_filtered*.npy file found in current directory")
input_path = input_files[0]

# Load memory-mapped array
print(f"Loading features from {input_path}")
X_arr = np.load(input_path, mmap_mode='r').astype(np.float32)

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
y_arr_regression = convert_labels(y_arr_classification, class_to_poles) * 1.0

n_tasks = X_arr.shape[1]

# First, split your original dataset (without augmentation) into train/validation
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
    X_arr, y_arr_regression, test_size=0.2, random_state=42
)

del X_arr, y_arr_regression, y_arr_classification
gc.collect()


# Function to augment the dataset by repeating each sample for each task,
# and adding a task identifier as an additional feature.
def augment_data(X, y):
    n_samples = X.shape[0]
    # Repeat X for each task (shape: [n_samples * n_tasks, n_features])
    X_aug = np.repeat(X, n_tasks, axis=0).astype(np.float32)
    # Create a column with task identifiers (0, 1, 2, …)
    task_ids = np.tile(np.arange(n_tasks), n_samples).reshape(-1, 1).astype(np.float32)
    # Append the task identifier to the features
    X_aug = np.hstack([X_aug, task_ids])
    
    # Flatten y so that each target value aligns with the corresponding task
    y_aug = y.flatten().astype(np.float32)
    return X_aug, y_aug


# Augment both training and validation datasets
# X_train_aug, y_train_aug = augment_data(X_train_orig, y_train_orig)
# X_val_aug, y_val_aug = augment_data(X_val_orig, y_val_orig)

# Create CatBoost Pools for training and validation
# train_pool = Pool(X_train_aug, y_train_aug)
# val_pool = Pool(X_val_aug, y_val_aug)

model = CatBoostRegressor(
    loss_function='RMSEWithUncertainty',
    verbose=100,
    iterations=1000,          # More reasonable iteration limit
    depth=6,                  # Reduce depth to control overfitting
    learning_rate=0.03,       # Explicit learning rate tuning
    l2_leaf_reg=5,            # Regularization for stability
    early_stopping_rounds=20, # Less aggressive early stopping
    random_seed=42,
)

batch_size = 10_000
n_train = X_train_orig.shape[0]

first_fit = True
for start in range(0, n_train, batch_size):
    end = min(start + batch_size, n_train)
    X_batch = X_train_orig[start:end].astype(np.float32)
    y_batch = y_train_orig[start:end].astype(np.float32)
    
    X_aug, y_aug = augment_data(X_batch, y_batch)
    train_pool = Pool(X_aug, y_aug)
    
    if first_fit:
        model.fit(train_pool)
        first_fit = False
    else:
        model.fit(train_pool, init_model=model, continue_training=True)
    
    del X_batch, y_batch, X_aug, y_aug, train_pool
    gc.collect()

# Fit the model
# model.fit(train_pool, eval_set=val_pool, use_best_model=True)
# Save model
# model_path = f"models/catboost_model_{d}_fold{fold}.cbm" if not raw else f"models/catboost_model_{d}_fold{fold}_raw.cbm"
# model.save_model(model_path)

# For prediction, we need to create an augmented validation set from the original features.
# Here, each original validation sample is repeated 3 times with the corresponding task id.
def prepare_prediction_data(X, n_tasks):
    n_samples = X.shape[0]
    X_pred = np.repeat(X, n_tasks, axis=0)
    task_ids = np.tile(np.arange(n_tasks), n_samples).reshape(-1, 1)
    X_pred = np.hstack([X_pred, task_ids])
    return X_pred

X_val_pred = prepare_prediction_data(X_val_orig, n_tasks)

# Get predictions with uncertainty. 
# The output shape is (n_val * n_tasks, 3), where:
# Column 0: mean prediction, Column 1: knowledge uncertainty, Column 2: data uncertainty.
pred_aug = model.virtual_ensembles_predict(
    X_val_pred,
    prediction_type='TotalUncertainty',
    virtual_ensembles_count=10
)

# Reshape the predictions into (n_val, n_tasks) for easier interpretation.
n_val = X_val_orig.shape[0]
preds_mean = pred_aug[:, 0].reshape(n_val, n_tasks)
knowledge_uncertainty = pred_aug[:, 1].reshape(n_val, n_tasks)
data_uncertainty = pred_aug[:, 2].reshape(n_val, n_tasks)

# Now, preds_mean contains your mean predictions for each of the 3 tasks,
# and uncertainty arrays are similarly reshaped.

# Post-process predictions: absolute value and rounding (as used in your evaluation).
preds_rounded = np.round(np.abs(preds_mean))

# Evaluate the final metric (ensure reconvert_labels is consistent).
accuracy = accuracy_score(
    reconvert_labels(preds_rounded, class_to_poles), 
    reconvert_labels(y_val_orig, class_to_poles)
)

print("Final accuracy:", accuracy)

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
    

print("Pipeline execution complete!")