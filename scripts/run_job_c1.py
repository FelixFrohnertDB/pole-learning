import numpy as np
import gc
import pickle
import glob
from sklearn.model_selection import train_test_split, KFold
from catboost import CatBoostClassifier

# Automatically find the matching file
input_files = glob.glob('training_features_128*.npy')
if not input_files:
    raise FileNotFoundError("No features_filtered*.npy file found in current directory")
input_path = input_files[0]

# Load memory-mapped array
print(f"Loading features from {input_path}")
X_arr = np.load(input_path, mmap_mode='r')

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

indices = np.arange(350_000)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

y_arr_classification = y_arr_classification[train_idx]
X_arr = X_arr[train_idx]


fold_accuracies = []

# Use k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(kf.split(X_arr)):
    print(f"  Fold {fold+1}/5")
    # Extract current fold data with selected feature count
    X_train, X_val = X_arr[train_index].astype(np.float32), X_arr[val_index].astype(np.float32)
    y_train, y_val = y_arr_classification[train_index], y_arr_classification[val_index]

    # Train model
    model = CatBoostClassifier(
        iterations=1000,
        verbose=100,
        early_stopping_rounds=10,
        random_seed=42+fold
    )

    model.fit(
        X_train, y_train, 
        eval_set=(X_val, y_val), 
        use_best_model=True,
        plot=False
    )

    # Save model
    model_path = f"models/catboost_model_c1_fold{fold}.cbm"
    model.save_model(model_path)
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    acc = np.sum(y_pred.flatten()==y_val)/np.sum(np.ones_like(y_val))
    fold_accuracies.append(acc)
    print(f"  Fold accuracy: {acc:.4f}")

    # Clean up
    del model, X_train, X_val, y_train, y_val
    gc.collect()

# Save fold accuracies
acc_file = f"fold_accuracies_c1.pkl" 
with open(acc_file, "wb") as f:
    pickle.dump(fold_accuracies, f)
    

print("Pipeline execution complete!")