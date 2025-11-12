"""
metrics.py

Evaluation metrics for seizure detection, including normalized AUPRC.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt


def normalized_auprc(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    """
    Compute AUPRC normalized by random baseline.
    
    This accounts for class imbalance - a random classifier would achieve
    AUPRC = proportion of positive class.
    
    Args:
        y_true: True labels [N]
        y_probs: Predicted probabilities for positive class [N]
        
    Returns:
        Normalized AUPRC in [0, 1], where:
        - 0.0 = random performance
        - 1.0 = perfect performance
    """
    auprc = average_precision_score(y_true, y_probs)
    baseline = y_true.mean()
    
    # Handle edge cases
    if baseline >= 0.999:  # Almost all positive
        return 1.0 if auprc >= 0.999 else 0.0
    if baseline <= 0.001:  # Almost all negative
        return 1.0 if auprc >= 0.999 else 0.0
    
    normalized = (auprc - baseline) / (1 - baseline)
    return max(0.0, min(1.0, normalized))  # Clip to [0, 1]


def compute_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        metric: 'f1', 'f2', or 'youden' (sensitivity + specificity - 1)
        
    Returns:
        optimal_threshold, optimal_metric_value
    """
    if metric == 'youden':
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        return thresholds[optimal_idx], youden_index[optimal_idx]
    
    # For F1/F2, search over thresholds
    thresholds = np.linspace(0, 1, 101)
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'f2':
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5,
    prefix: str = ''
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels [N]
        y_probs: Predicted probabilities for positive class [N]
        threshold: Classification threshold
        prefix: Prefix for metric names (e.g., 'train_', 'val_')
        
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_probs >= threshold).astype(int)
    
    # Baseline (for context)
    baseline = y_true.mean()
    
    # Threshold-independent metrics
    auprc_raw = average_precision_score(y_true, y_probs)
    auprc_norm = normalized_auprc(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)
    
    # Threshold-dependent metrics
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Clinical metrics
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    precision_pos = precision_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Compile metrics
    metrics = {
        f'{prefix}baseline': baseline,
        f'{prefix}auprc_raw': auprc_raw,
        f'{prefix}auprc_normalized': auprc_norm,
        f'{prefix}auroc': auroc,
        f'{prefix}f1': f1,
        f'{prefix}f2': f2,
        f'{prefix}mcc': mcc,
        f'{prefix}balanced_accuracy': balanced_acc,
        f'{prefix}sensitivity': sensitivity,
        f'{prefix}specificity': specificity,
        f'{prefix}precision': precision_pos,
        f'{prefix}threshold': threshold,
        f'{prefix}true_positives': int(tp),
        f'{prefix}false_positives': int(fp),
        f'{prefix}true_negatives': int(tn),
        f'{prefix}false_negatives': int(fn),
        f'{prefix}n_samples': len(y_true),
        f'{prefix}n_positive': int(y_true.sum()),
        f'{prefix}n_negative': int((1 - y_true).sum()),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Group metrics
    threshold_independent = ['auprc_raw', 'auprc_normalized', 'auroc', 'baseline']
    threshold_dependent = ['f1', 'f2', 'mcc', 'balanced_accuracy']
    clinical = ['sensitivity', 'specificity', 'precision', 'threshold']
    confusion = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
    
    # Remove prefix for printing
    clean_metrics = {}
    prefix = ''
    for key in metrics.keys():
        if '_' in key:
            parts = key.split('_', 1)
            if parts[0] in ['train', 'val', 'test']:
                prefix = parts[0] + '_'
                clean_metrics[parts[1]] = metrics[key]
            else:
                clean_metrics[key] = metrics[key]
        else:
            clean_metrics[key] = metrics[key]
    
    print("\nThreshold-Independent Metrics:")
    for metric in threshold_independent:
        if metric in clean_metrics:
            value = clean_metrics[metric]
            print(f"  {metric:20s}: {value:.4f}")
    
    print("\nThreshold-Dependent Metrics:")
    for metric in threshold_dependent:
        if metric in clean_metrics:
            value = clean_metrics[metric]
            print(f"  {metric:20s}: {value:.4f}")
    
    print("\nClinical Metrics:")
    for metric in clinical:
        if metric in clean_metrics:
            value = clean_metrics[metric]
            if metric == 'threshold':
                print(f"  {metric:20s}: {value:.4f}")
            else:
                print(f"  {metric:20s}: {value:.4f} ({value*100:.1f}%)")
    
    print("\nConfusion Matrix:")
    if all(m in clean_metrics for m in confusion):
        tp = clean_metrics['true_positives']
        fp = clean_metrics['false_positives']
        tn = clean_metrics['true_negatives']
        fn = clean_metrics['false_negatives']
        
        print(f"                Predicted")
        print(f"              Neg    Pos")
        print(f"  Actual Neg  {tn:5d}  {fp:5d}")
        print(f"         Pos  {fn:5d}  {tp:5d}")


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str = None
):
    """
    Plot ROC and Precision-Recall curves.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        save_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auroc = roc_auc_score(y_true, y_probs)
    
    axes[0].plot(fpr, tpr, linewidth=2, label=f'AUROC = {auroc:.3f}')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    baseline = y_true.mean()
    auprc_norm = normalized_auprc(y_true, y_probs)
    
    axes[1].plot(recall, precision, linewidth=2, 
                 label=f'AUPRC = {auprc:.3f}\nNormalized = {auprc_norm:.3f}')
    axes[1].axhline(baseline, color='k', linestyle='--', linewidth=1, 
                    label=f'Random = {baseline:.3f}')
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision (PPV)')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Curves saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Simulate imbalanced data
    n_samples = 1000
    n_positive = 100  # 10% imbalance
    
    y_true = np.array([1] * n_positive + [0] * (n_samples - n_positive))
    y_probs = np.random.rand(n_samples)
    y_probs[:n_positive] += 0.3  # Make positives slightly higher
    y_probs = np.clip(y_probs, 0, 1)
    
    # Compute metrics
    metrics = compute_comprehensive_metrics(y_true, y_probs, threshold=0.5)
    print_metrics(metrics, "Test Metrics")
    
    # Plot curves
    plot_roc_pr_curves(y_true, y_probs)