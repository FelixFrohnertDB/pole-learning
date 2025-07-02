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

# Map class index to individual position labels
print("Mapping class indices to individual pole positions...")

indices = np.arange(350_000)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

class_to_poles_arr = np.array([class_to_poles[i] for i in range(num_files)])
y_bt = class_to_poles_arr[y_arr_classification][:, 0] [train_idx]
y_bb = class_to_poles_arr[y_arr_classification][:, 1] [train_idx]
y_tb = class_to_poles_arr[y_arr_classification][:, 2] [train_idx]

X_arr = X_arr[train_idx]

del y_arr_classification 


targets = {'bt': y_bt, 'bb': y_bb, 'tb': y_tb}
fold_accuracies = {k: [] for k in targets}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(kf.split(X_arr)):
    print(f"Fold {fold + 1}/5")
    
    for pole, y_target in targets.items():
        print(f"  Training model for pole: {pole}")
        
        X_train = X_arr[train_index].astype(np.float32)
        X_val = X_arr[val_index].astype(np.float32)
        y_train = y_target[train_index]
        y_val = y_target[val_index]

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
        model_path = f"models/catboost_model_c3_{pole}_fold{fold}.cbm"
        model.save_model(model_path)

        # Predict and evaluate
        y_pred = model.predict(X_val).flatten()
        acc = np.sum(y_pred.flatten()==y_val)/np.sum(np.ones_like(y_val))
        fold_accuracies[pole].append(acc)
        print(f"    Accuracy [{pole}] = {acc:.4f}")

        # Cleanup
        del model, X_train, X_val, y_train, y_val
        gc.collect()

with open("fold_accuracies_c3.pkl", "wb") as f:
    pickle.dump(fold_accuracies, f)


