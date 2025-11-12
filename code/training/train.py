"""
train.py

Training utilities for seizure detection model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm

from metrics import compute_comprehensive_metrics, normalized_auprc


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' (higher is better) or 'min' (lower is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current validation score
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


class Trainer:
    """Trainer for seizure detection model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        grad_clip: float = 1.0,
        early_stopping_patience: int = 10
    ):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            grad_clip: Gradient clipping value
            early_stopping_patience: Patience for early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max'
        )
        
        # Track history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_auprc_normalized': [],
            'val_auprc_raw': [],
            'val_auroc': [],
            'learning_rates': []
        }
        
        self.best_val_score = 0
        self.best_epoch = 0
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for signals, labels, metadata in pbar:
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(signals)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * signals.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader.dataset)
        
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        train_auprc_norm = normalized_auprc(all_labels, all_probs)
        
        return {
            'train_loss': avg_loss,
            'train_auprc_normalized': train_auprc_norm
        }
    
    def validate(self) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Validate model.
        
        Returns:
            metrics: Dictionary of validation metrics
            all_labels: All true labels
            all_probs: All predicted probabilities
        """
        self.model.eval()
        
        total_loss = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for signals, labels, metadata in tqdm(self.val_loader, desc='Validation'):
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(signals)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                total_loss += loss.item() * signals.size(0)
                
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader.dataset)
        
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # Compute comprehensive metrics
        metrics = compute_comprehensive_metrics(
            all_labels,
            all_probs,
            threshold=0.5,
            prefix='val_'
        )
        
        metrics['val_loss'] = avg_loss
        
        return metrics, all_labels, all_probs
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = 'checkpoints',
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*60}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics, val_labels, val_probs = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_auprc_normalized'])
                else:
                    self.scheduler.step()
            
            # Track history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_auprc_normalized'].append(val_metrics['val_auprc_normalized'])
            self.history['val_auprc_raw'].append(val_metrics['val_auprc_raw'])
            self.history['val_auroc'].append(val_metrics['val_auroc'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"\nEpoch {epoch + 1} completed in {epoch_time:.1f}s")
                print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
                print(f"  Val Loss:   {val_metrics['val_loss']:.4f}")
                print(f"  Val AUPRC (norm):  {val_metrics['val_auprc_normalized']:.4f}")
                print(f"  Val AUPRC (raw):   {val_metrics['val_auprc_raw']:.4f}")
                print(f"  Val AUROC:         {val_metrics['val_auroc']:.4f}")
                print(f"  Val Baseline:      {val_metrics['val_baseline']:.4f}")
                print(f"  Val F2:            {val_metrics['val_f2']:.4f}")
                print(f"  Val Sensitivity:   {val_metrics['val_sensitivity']:.4f}")
                print(f"  Val Specificity:   {val_metrics['val_specificity']:.4f}")
                print(f"  Learning Rate:     {current_lr:.6f}")
            
            # Save best model
            val_score = val_metrics['val_auprc_normalized']
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                self.best_epoch = epoch + 1
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'history': self.history
                }
                
                if self.scheduler is not None:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
                
                torch.save(checkpoint, save_dir / 'best_model.pth')
                
                if verbose:
                    print(f"  → Best model saved (AUPRC norm: {val_score:.4f})")
            
            # Early stopping
            if self.early_stopping(val_score):
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    print(f"Best epoch was {self.best_epoch} with AUPRC norm: {self.best_val_score:.4f}")
                break
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training completed in {total_time/60:.1f} minutes")
            print(f"Best epoch: {self.best_epoch}")
            print(f"Best validation AUPRC (normalized): {self.best_val_score:.4f}")
            print(f"{'='*60}")
        
        return self.history
    
    def load_best_model(self, checkpoint_path: str):
        """Load best model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
        return checkpoint


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: 'adamw', 'adam', or 'sgd'
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        
    Returns:
        Optimizer
    """
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999))
        )
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999))
        )
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'onecycle',
    num_epochs: int = 50,
    steps_per_epoch: int = 100,
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: 'onecycle', 'cosine', 'step', or 'plateau'
        num_epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        
    Returns:
        Scheduler or None
    """
    if scheduler_name.lower() == 'onecycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr']),
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=kwargs.get('pct_start', 0.3),
            anneal_strategy='cos'
        )
    elif scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * steps_per_epoch,
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name.lower() == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    elif scheduler_name.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def create_criterion(
    criterion_name: str = 'cross_entropy',
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0
) -> nn.Module:
    """
    Create loss criterion.
    
    Args:
        criterion_name: 'cross_entropy' or 'focal'
        class_weights: Optional class weights for imbalanced data
        label_smoothing: Label smoothing factor
        
    Returns:
        Loss criterion
    """
    if criterion_name.lower() == 'cross_entropy':
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    elif criterion_name.lower() == 'focal':
        # Focal loss (custom implementation needed)
        raise NotImplementedError("Focal loss not implemented yet")
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")


if __name__ == "__main__":
    # Test training components
    from model import LightweightSeizureDetector
    from data_utils import create_dataloaders
    
    # Create dummy data loaders
    print("Testing training components...")
    
    # Create model
    model = LightweightSeizureDetector(base_filters=32)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = create_optimizer(model, 'adamw', learning_rate=1e-3)
    print(f"Optimizer: {type(optimizer).__name__}")
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, 'onecycle', num_epochs=10, steps_per_epoch=100)
    print(f"Scheduler: {type(scheduler).__name__}")
    
    # Create criterion
    criterion = create_criterion('cross_entropy', label_smoothing=0.1)
    print(f"Criterion: {type(criterion).__name__}")
    
    print("\nTraining components ready!")