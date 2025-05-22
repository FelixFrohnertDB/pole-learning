import numpy as np 
from scipy.optimize import minimize 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gc
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV
from tqdm import tqdm

import seaborn as sns
import pandas as pd 


plt.style.use('tableau-colorblind10')
def plot_style():
    font_size       = 12
    dpi             = 200

    params = {'figure.dpi': dpi,
              'savefig.dpi': dpi,
              'font.size': font_size,
              'font.family': "serif",
              'figure.titlesize': font_size,
              'legend.fontsize': font_size,
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'text.usetex': True,
             }

    plt.rcParams.update(params)
plot_style()

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

def get_class_from_output(outputs):
    # Convert outputs into a tuple of presence and count
    presence_bt, presence_bb, presence_tb = outputs[:,0], outputs[:,1], outputs[:,2]
    count_bt, count_bb, count_tb = outputs[:,3], outputs[:,4], outputs[:,5]
    
    # Convert continuous count values into discrete values (e.g., rounding)
    count_bt = np.round(count_bt.numpy())
    count_bb = np.round(count_bb.numpy())
    count_tb = np.round(count_tb.numpy())
    
    # Convert presence to binary (0 or 1)
    presence_bt = np.round(presence_bt.numpy())
    presence_bb = np.round(presence_bb.numpy())
    presence_tb = np.round(presence_tb.numpy())
    
    # Create a tuple of (presence_bt, presence_bb, presence_tb, count_bt, count_bb, count_tb)
    # We map this tuple into a class index
    class_mapping = {
        (1, 0, 0, 1, 0, 0): 0,  # Class 0
        (0, 1, 0, 0, 1, 0): 1,  # Class 1
        (0, 0, 1, 0, 0, 1): 2,  # Class 2
        (1, 1, 0, 1, 1, 0): 3,  # Class 3
        (0, 1, 1, 0, 1, 1): 4,  # Class 4
        (1, 1, 1, 1, 1, 1): 5,  # Class 5
        (0, 2, 0, 0, 2, 0): 6,  # Class 6
        (0, 1, 2, 0, 1, 2): 7   # Class 7
    }
    
    return class_mapping.get((presence_bt, presence_bb, presence_tb, count_bt, count_bb, count_tb), -1)  # -1 if no match

def compute_class_probabilities(lower, upper, class_boundaries):
    """Compute the probability of a prediction belonging to each class based on dynamic class boundaries."""
    n_classes = len(class_boundaries) + 1  # Total classes (one more than number of boundaries)
    class_probs = np.zeros((len(lower), n_classes))  # Stores probabilities for each sample

    for i in range(len(lower)):  # Loop through all samples
        probs = np.zeros(n_classes)  # Initialize probability vector

        # Class 0 (below the first boundary)
        probs[0] = max(0, min(1, (class_boundaries[0] - lower[i]) / (upper[i] - lower[i])))

        # Intermediate classes
        for j in range(1, len(class_boundaries)):
            lower_overlap = max(0, min(1, (class_boundaries[j - 1] - lower[i]) / (upper[i] - lower[i])))
            upper_overlap = max(0, min(1, (class_boundaries[j] - lower[i]) / (upper[i] - lower[i])))
            probs[j] = max(0, upper_overlap - lower_overlap)

        # Class N (above the last boundary)
        probs[-1] = max(0, min(1, (upper[i] - class_boundaries[-1]) / (upper[i] - lower[i])))

        # Normalize probabilities (avoid division by zero)
        total_prob = np.sum(probs)
        if total_prob > 0:
            class_probs[i] = probs / total_prob

    return class_probs.mean(axis=0)

def plot_uncer(preds, y_test, data_uncertainty, knowledge_uncertainty):
    fig, axes = plt.subplots(3, 3, figsize=(6, 4), sharey="row",  gridspec_kw={"wspace": 0, "hspace": 0}, constrained_layout=True)

    titles = ["BT", "BB", "TB"]

    # Data Uncertainty (Row 1)
    for i, ax in enumerate(axes[0]):
        sns.histplot(data_uncertainty[:, i], kde=True, bins=30, color='red', ax=ax)
        ax.set_title(titles[i], fontsize=10)
        if i == 0:
            ax.set_ylabel("Freq", fontsize=9)

    # Knowledge Uncertainty (Row 2)
    for i, ax in enumerate(axes[1]):
        sns.histplot(knowledge_uncertainty[:, i], kde=True, bins=30, color='blue', ax=ax)
        if i == 0:
            ax.set_ylabel("Freq", fontsize=9)

    
    for i, ax in enumerate(axes[2]):
    # Plot ideal diagonal
        ax.plot([y_test[:, i].min(), y_test[:, i].max()], 
                [y_test[:, i].min(), y_test[:, i].max()], 
                'k--', lw=1)

        if i == 0:
            ax.set_ylabel("Pred", fontsize=9)

        # Scatter plot of raw predictions (underscore marker)
        ax.plot(y_test[:, i], preds[:, i], "_k", alpha=0.5)

        # Loop over each unique integer value of y_test
        for y in np.unique(y_test[:, i]):
            subset_preds = preds[:, i][y_test[:, i] == y]

            

            # Mean & std error bar
            ax.errorbar(
                y, np.mean(subset_preds),
                yerr=np.std(subset_preds),
                fmt="x", color="r", markersize=6, capsize=3
            )

        # Make sure x-ticks align with integer y_test values
        ax.set_xticks(np.unique(y_test[:, i]))
        ax.set_xlim(y_test[:, i].min() - 0.5, y_test[:, i].max() + 0.5)


def plot_uncer_against_exact(preds, y_test, data_uncertainty):
    titles = ["BT", "BB", "TB"]

    fig, axes = plt.subplots(1, 3, figsize=(6, 2), sharey=True, sharex=True, gridspec_kw={"wspace": 0, "hspace": 0}, constrained_layout=True)

    for i, ax in enumerate(axes):
        ax.scatter(data_uncertainty[:, i], (y_test[:, i] - preds[:, i])**2, alpha=0.7, s=5)
        ax.plot([1e-5, 1], [1e-5, 1], 'k--', lw=1)  # Ideal line
        ax.set_title(titles[i], fontsize=9)
        ax.set_yscale("log")
        ax.set_xscale("log")

    # Shared labels
    fig.text(0.5, -0.02, "Model Uncertainty", ha="center", fontsize=10)
    fig.text(-0.02, 0.5, "Mean Squared Error", va="center", rotation="vertical", fontsize=10)

    plt.show()

def plot_predictions_with_confidence(df,):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    targets = ["BT", "BB", "TB"]
    # Plot half-violins for predictions
    sns.violinplot(
        x="Predicted", 
        y="Target", 
        data=df, 
        inner=None, 
        linewidth=1, 
        orient="h", 
        cut=0, 
        color="red", 
        width=0.7
    )

    # Add scatter points for individual predictions
    # sns.stripplot(
    #     x="Predicted", 
    #     y="Target", 
    #     data=df, 
    #     color="black", 
    #     size=2, 
    #     alpha=0.5
    # )

    # Add error bars for confidence intervals
    for i, target in enumerate(targets):
        subset = df[df["Target"] == target]
        lower = subset["Lower Bound"].mean()
        upper = subset["Upper Bound"].mean()
        pred_mean = subset["Predicted"].mean()
        
        ax.plot([lower, upper], [i, i], color="purple", lw=2, marker="|")  # CI as a horizontal bar
        ax.scatter(pred_mean, i, color="black", zorder=3)  # Mean point

    ax.set_xlabel("Predicted Value")
    # ax.set_title("Predicted Distributions with Confidence Bounds")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.show()

def plot_cm(y_true, y_pred, class_to_poles):

    custom_labels = [str(class_to_poles[i]) for i in range(len(class_to_poles))]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(np.unique(y_true).shape[0]))

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(np.unique(y_true).shape[0]))
    fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size if needed
    disp.plot(ax=ax)

    # Update x and y axis tick labels with the custom labels
    ax.set_xticks(np.arange(len(custom_labels)))
    ax.set_yticks(np.arange(len(custom_labels)))
    ax.set_xticklabels(custom_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(custom_labels, fontsize=10)

    # Adjust layout for readability
    plt.tight_layout()
    plt.show()

    print("Acc:",accuracy_score(y_true,y_pred))