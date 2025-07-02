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

slurm_inx = int(sys.argv[2])
print(f"Slurm: {slurm_inx}")

_feature_names = glob.glob('feature_names*.npy')

# Extract the part between the last underscore and .npy
identifier = _feature_names[0].split('_')[-1].replace('.npy', '')

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


# Step 6: Reload reordered array with memory-mapping
print("Loading sorted features...")
if raw:
    n_samples = 350000
    n_features = 75
    
    output_path = f'full_features_sorted_raw_{n_samples}_{n_features}_{identifier}.npy'
    X_arr = np.memmap(output_path, dtype='float32', mode='r', shape=(n_samples, n_features))
else:

    n_samples = 350000
    n_features = 423 #429 439 or 423
    
    output_path = f'full_features_sorted_{n_samples}_{n_features}_{identifier}.npy'
    X_arr = np.memmap(output_path, dtype='float32', mode='r', shape=(n_samples, n_features))



print("Starting cross-validation with different feature counts...")
# Define feature counts to test using log-spaced percentages
feature_percentages = np.logspace(-2, 0, 5)[-1]  # From 1% to 100% of features
feature_counts = np.unique(
    np.clip((feature_percentages * X_arr.shape[1]).astype(int), 1, X_arr.shape[1])
)

# Loop through different feature counts
for d in feature_counts:
    print(f"\nEvaluating with top {d} features ({d/X_arr.shape[1]*100:.2f}% of total)")
    fold_accuracies = []

    # Use k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(X_arr)):
        print(f"  Fold {fold+1}/5")
        if fold == slurm_inx:
            
            # Extract current fold data with selected feature count
            X_train, X_val = X_arr[train_index][:, :d], X_arr[val_index][:, :d]
            y_train, y_val = y_arr_classification[train_index], y_arr_classification[val_index]

            # Train model
            model = CatBoostClassifier(
                iterations=1000,
                verbose=50,
                early_stopping_rounds=10,
                random_seed=42+fold,
            )

            model.fit(
                X_train, y_train, 
                eval_set=(X_val, y_val), 
                use_best_model=True,
                plot=False
            )

            # Save model
            model_path = f"models/catboost_model_{identifier}_{d}_fold{fold}.cbm" if not raw else f"models/catboost_model_{identifier}_{d}_fold{fold}_raw.cbm"
            model.save_model(model_path)
            print(f"  Model saved to {model_path}")

            # Predict and evaluate
            y_pred = np.abs(np.round(model.predict(X_val)))
            acc = accuracy_score(model.predict(X_val),y_val)
            fold_accuracies.append(acc)
            print(f"  Fold accuracy: {acc:.4f}")

            # Clean up
            del model, X_train, X_val, y_train, y_val
            gc.collect()

    # Save fold accuracies
    # acc_file = f"fold_accuracies_{d}_{identifier}.pkl" if not raw else f"fold_accuracies_{d}_raw_{identifier}.pkl"
    # with open(acc_file, "wb") as f:
    #     pickle.dump({d: fold_accuracies}, f)
    

print("Pipeline execution complete!")