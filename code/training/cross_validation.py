"""
cross_validation.py

Cross-validation and hyperparameter optimization using Optuna.
"""

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from model import LightweightSeizureDetector, count_parameters
from data_utils import SeizureDataset, create_dataloaders
from train import Trainer, create_optimizer, create_scheduler, create_criterion
from metrics import compute_comprehensive_metrics, normalized_auprc, print_metrics


def create_cv_splits(
    dataset: SeizureDataset,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    Create patient-stratified cross-validation splits.
    
    Args:
        dataset: SeizureDataset
        n_splits: Number of CV folds
        random_state: Random seed
        
    Returns:
        List of (train_patient_ids, val_patient_ids) tuples
    """
    # Get all patient IDs and their labels
    patient_ids = dataset.get_patient_ids()
    
    # Get patient-level labels (use majority class per patient)
    patient_labels = {}
    for patient_id in patient_ids:
        patient_samples = [
            s for s in dataset.sample_index 
            if s['patient_id'] == patient_id
        ]
        
        # Get labels for this patient
        labels = []
        with dataset.h5_path.open('rb') as f:
            import h5py
            h5f = h5py.File(dataset.h5_path, 'r')
            for sample in patient_samples:
                label = int(h5f[sample['patient_id']][sample['seizure_id']]['labels'][sample['sample_idx']])
                if dataset.combine_onset_spread and label > 0:
                    label = 1
                labels.append(label)
            h5f.close()
        
        # Majority vote for stratification
        patient_labels[patient_id] = int(np.median(labels))
    
    # Create stratified group k-fold
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Dummy data for splitting
    X_dummy = np.arange(len(patient_ids))
    y_stratify = np.array([patient_labels[pid] for pid in patient_ids])
    groups = np.array(patient_ids)
    
    # Generate splits
    cv_splits = []
    for train_idx, val_idx in sgkf.split(X_dummy, y_stratify, groups=groups):
        train_patients = [patient_ids[i] for i in train_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        cv_splits.append((train_patients, val_patients))
        
        print(f"Fold {len(cv_splits)}: Train={len(train_patients)} patients, Val={len(val_patients)} patients")
    
    return cv_splits


def objective(
    trial: Trial,
    h5_path: str,
    cv_splits: List[Tuple[List[str], List[str]]],
    config: Dict
) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial
        h5_path: Path to HDF5 data file
        cv_splits: List of (train_patients, val_patients) tuples
        config: Configuration dictionary
        
    Returns:
        Mean cross-validation normalized AUPRC
    """
    # Sample hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    base_filters = trial.suggest_categorical('base_filters', [16, 24, 32, 48])
    dropout_blocks = trial.suggest_float('dropout_blocks', 0.1, 0.4)
    dropout_head = trial.suggest_float('dropout_head', 0.2, 0.5)
    num_blocks = trial.suggest_int('num_blocks', 2, 4)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)
    
    # Scheduler parameters
    scheduler_type = trial.suggest_categorical('scheduler', ['onecycle', 'cosine', 'plateau'])
    if scheduler_type == 'onecycle':
        pct_start = trial.suggest_float('pct_start', 0.2, 0.4)
    else:
        pct_start = 0.3
    
    # Dilation pattern
    if num_blocks == 2:
        dilations = [2, 4]
    elif num_blocks == 3:
        dilations = [2, 4, 8]
    else:  # num_blocks == 4
        dilations = [2, 4, 8, 16]
    
    # Track fold metrics
    fold_metrics = []
    
    # Cross-validation loop
    for fold_idx, (train_patients, val_patients) in enumerate(cv_splits):
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} - Fold {fold_idx + 1}/{len(cv_splits)}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_loader, val_loader = create_dataloaders(
            h5_path,
            train_patients,
            val_patients,
            batch_size=batch_size,
            num_workers=config.get('num_workers', 4)
        )
        
        # Calculate class imbalance for this fold
        train_dataset = train_loader.dataset
        train_dist = train_dataset.get_class_distribution()
        
        # Create class weights
        if config.get('use_class_weights', True):
            n_samples = sum(train_dist.values())
            n_classes = len(train_dist)
            class_weights = torch.tensor([
                n_samples / (n_classes * train_dist[i]) 
                for i in range(n_classes)
            ], dtype=torch.float32).cuda()
        else:
            class_weights = None
        
        # Create model
        model = LightweightSeizureDetector(
            num_classes=2,
            base_filters=base_filters,
            num_blocks=num_blocks,
            dilations=dilations,
            dropout_blocks=dropout_blocks,
            dropout_head=dropout_head
        )
        
        if fold_idx == 0:
            print(f"Model parameters: {count_parameters(model):,}")
        
        # Create optimizer, scheduler, criterion
        optimizer = create_optimizer(
            model,
            optimizer_name='adamw',
            learning_rate=lr,
            weight_decay=weight_decay
        )
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_name=scheduler_type,
            num_epochs=config['max_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=pct_start
        )
        
        criterion = create_criterion(
            'cross_entropy',
            class_weights=class_weights,
            label_smoothing=label_smoothing
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.get('device', 'cuda'),
            grad_clip=config.get('grad_clip', 1.0),
            early_stopping_patience=config.get('early_stopping_patience', 5)
        )
        
        # Train
        history = trainer.train(
            num_epochs=config['max_epochs'],
            save_dir=f"checkpoints/trial_{trial.number}_fold_{fold_idx}",
            verbose=False
        )
        
        # Get best validation metrics
        best_auprc_norm = trainer.best_val_score
        
        # Load best model and evaluate
        checkpoint = trainer.load_best_model(
            f"checkpoints/trial_{trial.number}_fold_{fold_idx}/best_model.pth"
        )
        
        val_metrics = checkpoint['val_metrics']
        
        fold_metrics.append({
            'auprc_normalized': val_metrics['val_auprc_normalized'],
            'auprc_raw': val_metrics['val_auprc_raw'],
            'auroc': val_metrics['val_auroc'],
            'f2': val_metrics['val_f2'],
            'sensitivity': val_metrics['val_sensitivity'],
            'specificity': val_metrics['val_specificity'],
            'baseline': val_metrics['val_baseline']
        })
        
        print(f"\nFold {fold_idx + 1} Results:")
        print(f"  Baseline:      {val_metrics['val_baseline']:.4f}")
        print(f"  AUPRC (raw):   {val_metrics['val_auprc_raw']:.4f}")
        print(f"  AUPRC (norm):  {val_metrics['val_auprc_normalized']:.4f}")
        print(f"  AUROC:         {val_metrics['val_auroc']:.4f}")
        print(f"  F2:            {val_metrics['val_f2']:.4f}")
        
        # Report intermediate values for pruning
        trial.report(best_auprc_norm, fold_idx)
        
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Aggregate metrics across folds
    mean_metrics = {
        metric: np.mean([f[metric] for f in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    std_metrics = {
        metric: np.std([f[metric] for f in fold_metrics])
        for metric in fold_metrics[0].keys()
    }
    
    # Log all metrics as user attributes
    for metric, value in mean_metrics.items():
        trial.set_user_attr(f'mean_{metric}', value)
        trial.set_user_attr(f'std_{metric}', std_metrics[metric])
    
    trial.set_user_attr('fold_baselines', [f['baseline'] for f in fold_metrics])
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number} Summary:")
    print(f"{'='*60}")
    print(f"Mean Baseline:        {mean_metrics['baseline']:.4f}")
    print(f"Mean AUPRC (raw):     {mean_metrics['auprc_raw']:.4f} ± {std_metrics['auprc_raw']:.4f}")
    print(f"Mean AUPRC (norm):    {mean_metrics['auprc_normalized']:.4f} ± {std_metrics['auprc_normalized']:.4f}")
    print(f"Mean AUROC:           {mean_metrics['auroc']:.4f} ± {std_metrics['auroc']:.4f}")
    print(f"Mean F2:              {mean_metrics['f2']:.4f} ± {std_metrics['f2']:.4f}")
    print(f"Mean Sensitivity:     {mean_metrics['sensitivity']:.4f} ± {std_metrics['sensitivity']:.4f}")
    print(f"Mean Specificity:     {mean_metrics['specificity']:.4f} ± {std_metrics['specificity']:.4f}")
    
    # Return mean normalized AUPRC (primary metric)
    return mean_metrics['auprc_normalized']


def run_hyperparameter_optimization(
    h5_path: str,
    config: Dict,
    n_trials: int = 50,
    study_name: str = 'seizure_detection'
) -> optuna.Study:
    """
    Run hyperparameter optimization.
    
    Args:
        h5_path: Path to HDF5 data file
        config: Configuration dictionary
        n_trials: Number of trials to run
        study_name: Name of study
        
    Returns:
        Optuna study object
    """
    # Create dataset to get patient IDs
    dataset = SeizureDataset(h5_path, combine_onset_spread=True)
    
    # Create CV splits
    print("Creating cross-validation splits...")
    cv_splits = create_cv_splits(
        dataset,
        n_splits=config.get('n_folds', 5),
        random_state=config.get('random_state', 42)
    )
    
    print(f"\nStarting hyperparameter optimization with {n_trials} trials...")
    print(f"Each trial will be evaluated on {len(cv_splits)} folds")
    print(f"Primary metric: Normalized AUPRC")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=config.get('random_state', 42)),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, h5_path, cv_splits, config),
        n_trials=n_trials,
        n_jobs=1,  # Sequential (each trial uses GPU)
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*60)
    
    best_trial = study.best_trial
    print(f"\nBest Trial: {best_trial.number}")
    print(f"Best Normalized AUPRC: {best_trial.value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key:20s}: {value}")
    
    print(f"\nBest Trial Metrics:")
    print(f"  Raw AUPRC:     {best_trial.user_attrs['mean_auprc_raw']:.4f}")
    print(f"  AUROC:         {best_trial.user_attrs['mean_auroc']:.4f}")
    print(f"  F2:            {best_trial.user_attrs['mean_f2']:.4f}")
    print(f"  Sensitivity:   {best_trial.user_attrs['mean_sensitivity']:.4f}")
    print(f"  Specificity:   {best_trial.user_attrs['mean_specificity']:.4f}")
    print(f"  Baseline:      {best_trial.user_attrs['mean_baseline']:.4f}")
    
    # Save study
    save_dir = Path(config.get('output_dir', 'results'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best hyperparameters
    best_params = {
        'hyperparameters': best_trial.params,
        'metrics': {
            'normalized_auprc': best_trial.value,
            **{k: v for k, v in best_trial.user_attrs.items()}
        }
    }
    
    with open(save_dir / 'best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"\nBest hyperparameters saved to {save_dir / 'best_hyperparameters.json'}")
    
    # Plot optimization history
    try:
        import optuna.visualization as vis
        import matplotlib.pyplot as plt
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_image(str(save_dir / 'optimization_history.png'))
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_image(str(save_dir / 'param_importances.png'))
        
        # Parallel coordinate plot
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(str(save_dir / 'parallel_coordinate.png'))
        
        print(f"Visualization plots saved to {save_dir}/")
        
    except Exception as e:
        print(f"Could not create visualizations: {e}")
        print("Install plotly and kaleido for visualizations: pip install plotly kaleido")
    
    return study


if __name__ == "__main__":
    # Example configuration
    config = {
        'max_epochs': 30,
        'n_folds': 5,
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'grad_clip': 1.0,
        'early_stopping_patience': 5,
        'use_class_weights': True,
        'random_state': 42,
        'output_dir': 'results'
    }
    
    # Run optimization
    study = run_hyperparameter_optimization(
        h5_path='seizure_data.h5',
        config=config,
        n_trials=50,
        study_name='seizure_detection_cv'
    )