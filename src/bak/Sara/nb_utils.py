### Imports 

import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib') # adapt

import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter
from cuda_selector import auto_cuda
from itertools import combinations, product
from matplotlib.lines import Line2D
from numpy.random import randint
from pathlib import Path as Path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score as auc
from time import time
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm

### Functions

def accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)  # get the class index with the highest probability
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def plot_confusion_matrix(conf_mat):
    if conf_mat.shape[0]>10:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=(20, 20))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    n_classes = conf_mat.shape[0]
    tick_marks = np.arange(n_classes)
    # plt.xticks(tick_marks, [f"Class {i}" for i in range(n_classes)], rotation=45)
    # plt.yticks(tick_marks, [f"Class {i}" for i in range(n_classes)])
    plt.xticks(tick_marks, [f"{i}" for i in range(n_classes)], rotation=45)
    plt.yticks(tick_marks, [f"{i}" for i in range(n_classes)])
    
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, str(conf_mat[i, j]),
                     horizontalalignment="center",
                     color="white" if conf_mat[i, j] > conf_mat.max() / 2. else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def eval_acc(df_sorted, target_acc):
    '''
    df_sorted: pandas.DataFrame with columns ['score', 'result'] sorted with 'score' in descending order
    target_acc: value of the target accuracy

    Returns:
    num_to_remove: points to remove to reach the target accuracy
    '''
    num_to_remove = 0
    for i in range(len(df_sorted)):
        df_subset = df_sorted.iloc[:len(df_sorted) - num_to_remove - 1]
        current_acc = df_subset['result'].sum() / len(df_subset)
        
        if current_acc >= target_acc:
            break
        num_to_remove += 1
    return num_to_remove

def get_acc_removed(df_sorted, target_acc, eps=1e-4):
    accuracy = []
    num_removed = []
    found = False
    
    for num in range(len(df_sorted)):
        df_removed = df_sorted.iloc[:-num]
        acc = df_removed['result'].sum() / len(df_removed)
        if target_acc - eps < acc and acc > target_acc + eps:
            if not found:
                r_ = num / len(df_sorted)
                found = True
        accuracy.append(acc)
        num_removed.append(num)

    return np.array(accuracy), np.array(num_removed), r_

def get_best_config_from_df(df, score_type, cls_type, layer):
    row = df.query("score_type == @score_type and cls_type == @cls_type and layer == @layer")
    if not row.empty:
        peep_size, n_cls = ast.literal_eval(row.iloc[0]['best_config'])
        return peep_size, n_cls
    else:
        return None

def get_indices_to_retain_or_reject(df_sorted, num_to_remove):
    '''
    From the sorted DataFrame, compute the indices of the retained and rejected inputs.
    '''
    rejected_indices = df_sorted.iloc[-num_to_remove:].index.tolist()
    retained_indices = df_sorted.iloc[:-num_to_remove].index.tolist()
    return retained_indices, rejected_indices

def confidence_score(dnn_output, ph_sel, intermediate_layer, score_type='max', cls_type='tKMeans', criterion='max'):
    '''
    Computes confidence scores based on specified criteria, obtained by combining the output of a DNN
    and intermediate layer peepholes.

    Parameters:
    ----------
    dnn_output : array-like or np.ndarray
        The output from the DNN, expected as a NumPy array or convertible to one.
    
    ph_sel : dict
        A nested dictionary containing selection parameters for the intermediate layer activations.
        It must follow the structure: `ph_sel[score_type][cls_type][intermediate_layer]`.
    
    intermediate_layer : str or any hashable type
        The key used to retrieve the desired intermediate layer activations from `ph_sel`.
    
    score_type : str, optional (default='max')
        The key used to select the scoring type within `ph_sel`.
    
    cls_type : str, optional (default='tKMeans')
        The key used to select the classification type within `ph_sel`.

    criterion : str, optional (default='max')
        The method to compute confidence vectors. Choices are:
        - 'max': Element-wise maximum of intermediate layer activations and DNN output.
        - 'min': Element-wise minimum of intermediate layer activations and DNN output.
        - 'product': Element-wise product of intermediate layer activations and DNN output.

    Returns:
    -------
    np.ndarray
        An array of confidence scores for each data point.

    Raises:
    -------
    ValueError
        If the required keys in `ph_sel` are not present or if inputs cannot be converted to NumPy arrays.

    Examples:
    --------
    >>> dnn_output = np.array([[0.2, 0.8], [0.4, 0.6]])
    >>> ph_sel = {'max': {'tKMeans': {'layer1': [[0.1, 0.9], [0.3, 0.7]]}}}
    >>> confidence_score(dnn_output, ph_sel, 'layer1')
    array([0.9, 0.7])
    '''
    
    # ensure inputs are NumPy arrays
    try:
        dnn_output = np.array(dnn_output, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"dnn_output cannot be converted to a NumPy array: {e}")
    
    try:
        int_layer = np.array(ph_sel[score_type][cls_type][intermediate_layer], dtype=np.float64)
    except KeyError as e:
        raise ValueError(f"Invalid key in ph_sel: {e}")
    except Exception as e:
        raise ValueError(f"ph_sel[{score_type}][{cls_type}][{intermediate_layer}] cannot be converted to a NumPy array: {e}")

    conf_vectors = []

    for i, o in zip(int_layer, dnn_output):
        if criterion == 'max':
            v = np.max((i, o), axis=0)
        elif criterion == 'min':
            v = np.min((i, o), axis=0)
        elif criterion == 'product':
            v = i * o
        else:
            raise ValueError(f"Invalid criterion: {criterion}. Expected 'max', 'min', or 'product'.")
        
        # normalize the vector
        v = v / np.sum(v)
        conf_vectors.append(v)

    conf_vectors = np.array(conf_vectors)
    conf_score = np.max(conf_vectors, axis=1)

    return conf_score


def compute_normalized_joint_entropy(prob_matrix, log_base=2, adjusted=True):
    """
    Computes the normalized joint entropy of a 2D matrix treated as a joint probability distribution.
    The entropy is normalized to the range [0, 1].

    Parameters:
    ----------
    prob_matrix : np.ndarray
        A 2D NumPy array representing joint probabilities across rows and columns.
        The elements must sum to 1 (or will be normalized to sum to 1).
    log_base : int, optional (default=2)
        The base of the logarithm used for entropy calculation.
    adjusted : bool, optional (default=True)
        Whether to adjust entropy of the probability vector, scaled to [0, 1], where
        1 is the best (high uncertainty) and 0 is the worst (low uncertainty).

    Returns:
    -------
    float
        The normalized joint entropy of the system (scaled to [0, 1]).
    """
    # ensure valid probabilities
    if not np.all(prob_matrix >= 0):
        raise ValueError("All elements in prob_matrix must be non-negative.")
    
    # normalize to ensure total sum is 1
    prob_matrix = prob_matrix / np.sum(prob_matrix)

    # choose logarithm function
    if log_base == 2:
        log_fn = np.log2
    elif log_base == np.e:
        log_fn = np.log
    elif log_base == 10:
        log_fn = np.log10
    else:
        raise ValueError("Unsupported log_base. Use 2, np.e, or 10.")

    joint_entropy = -np.sum(prob_matrix * log_fn(prob_matrix + 1e-12))  # Avoid log(0)

    # maximum entropy for a uniform distribution (all entries equal)
    n, m = prob_matrix.shape  
    max_entropy = log_fn(n * m)

    # normalize o the range [0, 1]
    normalized_joint_entropy = joint_entropy / max_entropy
    if adjusted:
        normalized_joint_entropy = 1 - normalized_joint_entropy

    return normalized_joint_entropy


def normalized_entropy(prob_vector, log_base=2, adjusted=True):
    """
    Computes the normalized entropy of a probability vector, scaled to [0, 1].

    Parameters:
    ----------
    prob_vector : np.ndarray
        A 1D array of probabilities that sum to 1.
    log_base : int, optional (default=2)
        The base of the logarithm used for entropy calculation.
    adjusted : bool, optional (default=True)
        Whether to adjust entropy of the probability vector, scaled to [0, 1], where
        1 is the best (high uncertainty) and 0 is the worst (low uncertainty).

    Returns:
    -------
    float
        The normalized entropy of the probability vector.
    """
    # ensure valid probabilities
    if not np.all((prob_vector >= 0) & (prob_vector <= 1)):
        raise ValueError("All elements in prob_vector must be in the range [0, 1].")
    if not np.isclose(np.sum(prob_vector), 1.0):
        raise ValueError("The elements of prob_vector must sum to 1.")

    # choose logarithm function
    if log_base == 2:
        log_fn = np.log2
    elif log_base == np.e:
        log_fn = np.log
    elif log_base == 10:
        log_fn = np.log10
    else:
        raise ValueError("Unsupported log_base. Use 2, np.e, or 10.")

    # compute Shannon entropy
    entropy = -np.sum(prob_vector * log_fn(prob_vector + 1e-12))  # avoid log(0)

    # maximum entropy for a uniform distribution
    n = len(prob_vector)
    max_entropy = log_fn(n)

    # normalize entropy
    normalized_entropy = entropy / max_entropy
    if adjusted:
        normalized_entropy = 1 - normalized_entropy

    return normalized_entropy

def plot_results(axs, row, col, title, acc_vs_rej, score_key, target_acc, x_values, correct_mask, linestyle='-', label_suffix='', hist=True):
    """
    A helper function to plot accuracy vs rejection and histograms.

    Args:
    axs: Subplots axis array.
    row, col: Indices for the subplot.
    title: Title for the plot.
    acc_vs_rej: Dictionary containing accuracy vs rejection data.
    score_key: Key to access the specific score in acc_vs_rej.
    target_acc: Target accuracy line value.
    x_values: Array of score values for histograms.
    correct_mask: Boolean mask for correct predictions.
    linestyle: Line style for the plots.
    label_suffix: Optional suffix to distinguish labels.
    hist: Optional whether to plot also histograms.
    """
    axs[row, col].set_title(title, fontsize=9)
    axs[row, col].plot(acc_vs_rej[score_key]['rej'], acc_vs_rej[score_key]['acc'], label=f'{score_key} {label_suffix}', lw=1, ls=linestyle)
    axs[row, col].axhline(target_acc, color='red', linestyle='--', lw=1, label='target accuracy')

    if hist:
        bins = np.linspace(x_values.min(), x_values.max(), 50)
        axs[row, col + 1].hist(x_values[correct_mask], bins=bins, density=True, log=True, alpha=0.5, label='correct')
        axs[row, col + 1].hist(x_values[~correct_mask], bins=bins, density=True, log=True, alpha=0.5, label='wrong')

def compute_reliability_and_ece(df_labels, df_scores, score_type, num_bins=10):
    """
    Compute reliability metrics and Expected Calibration Error (ECE).
    
    Parameters:
        df_labels (pd.DataFrame):
        df_scores (pd.DataFrame):
        score_type (str):
        num_bins (int): Number of bins to use for confidence intervals.
        
    Returns:
        dict: A dictionary containing:
            - "bin_centers": Midpoints of the confidence bins.
            - "bin_accuracy": Accuracy for each bin.
            - "bin_confidence": Average confidence for each bin.
            - "bin_counts": Number of data points in each bin.
            - "ece": Expected Calibration Error (ECE).
    """
    
    true_labels = df_labels['true'].values 
    predicted_labels = df_labels['label'].values # dnn prediction

    confidence = df_scores[score_type].values 

    correct_predictions = (true_labels == predicted_labels).astype(int)

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(confidence, bin_edges, right=True)

    bin_accuracy = []
    bin_confidence = []
    bin_counts = []

    for i in range(1, len(bin_edges)):
        bin_mask = bin_indices == i
        bin_counts.append(bin_mask.sum())
        if bin_mask.any():
            bin_accuracy.append(correct_predictions[bin_mask].mean())
            bin_confidence.append(confidence[bin_mask].mean())
        else:
            bin_accuracy.append(0)
            bin_confidence.append(0)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 

    # compute ECE
    total_points = len(confidence)
    ece = 0
    for i in range(len(bin_accuracy)):
        weight = bin_counts[i] / total_points
        ece += weight * abs(bin_accuracy[i] - bin_confidence[i])

    return {
        "bin_centers": bin_centers,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
        "ece": ece
    }

def map_labels_(labels, mapping_dict):
    '''
    For superclasses mapping.
    '''
    mapped_labels = []
    for label in labels:
        for new, old in mapping_dict.items():
            if label in old:
                mapped_labels.append(new)
    # return torch.tensor(mapped_labels)
    return mapped_labels

def compute_intersections(idx_dict_subset):
    """
    Computes intersections between all layers in idx_dict_subset.
    Returns a dictionary of intersections.
    """
    intersection_dict = {}
    layers = list(idx_dict_subset.keys())
    
    for layer_pair in combinations(layers, 2):
        layer1, layer2 = layer_pair
        intersection_dict[layer_pair] = set(idx_dict_subset[layer1]).intersection(idx_dict_subset[layer2])
    
    return intersection_dict

def intersections_to_df(intersection_dict):
    """
    Converts the intersections dictionary to a DataFrame.
    Columns: Layer 1, Layer 2, Intersection Size, Intersected Indices
    """
    data = []
    for (layer1, layer2), intersection in intersection_dict.items():
        data.append({
            "Layer 1": layer1,
            "Layer 2": layer2,
            "Intersection Size": len(intersection),
            "Intersected Indices": list(intersection)
        })
    return pd.DataFrame(data)


def compute_all_combinations_intersections(idx_dict_subset):
    """
    Computes intersections between all combinations of layers in idx_dict_subset.
    Returns a dictionary of intersections.
    """
    intersection_dict = {}
    layers = list(idx_dict_subset.keys())
    n_layers = len(layers)
    
    for r in range(2, n_layers + 1):
        for layer_combination in combinations(layers, r):
            intersection_dict[layer_combination] = set.intersection(
                *(set(idx_dict_subset[layer]) for layer in layer_combination)
            )
    
    return intersection_dict

def intersections_to_df_all_combinations(intersection_dict):
    """
    Converts the intersections dictionary to a DataFrame for all combinations.
    Columns: Layers, Combination Size, Intersection Size, Intersected Indices
    """
    data = []
    for layers, intersection in intersection_dict.items():
        data.append({
            "Layers": layers,
            "Combination Size": len(layers),
            "Intersection Size": len(intersection),
            "Intersected Indices": list(intersection)
        })
    return pd.DataFrame(data)
