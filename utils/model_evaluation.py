"""
Utility functions for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix,
    classification_report
)


def evaluate_regression_model(y_true, y_pred, model_name="Model"):
    """
    Evaluate a regression model using various metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, default="Model"
        Name of the model for display purposes
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.show()
    
    # Plot residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residuals Plot')
    plt.show()
    
    # Residuals distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title(f'{model_name}: Residuals Distribution')
    plt.show()
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_classification_model(y_true, y_pred, y_proba=None, model_name="Model", average='binary'):
    """
    Evaluate a classification model using various metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values (class labels)
    y_proba : array-like, default=None
        Predicted probabilities for the positive class (for binary classification)
        or for all classes (for multiclass)
    model_name : str, default="Model"
        Name of the model for display purposes
    average : str, default='binary'
        Averaging method for multiclass/multilabel classification
        ('binary', 'micro', 'macro', 'weighted')
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Handle binary vs multiclass
    is_binary = len(np.unique(y_true)) == 2
    
    if is_binary and average == 'binary':
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    else:
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        f1 = f1_score(y_true, y_pred, average=average)
    
    # Print metrics
    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name}: Confusion Matrix')
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # ROC and PR curves (for binary classification)
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    if is_binary and y_proba is not None:
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            # Get proba for positive class if we have probabilities for all classes
            y_proba = y_proba[:, 1]
            
        roc_auc = roc_auc_score(y_true, y_proba)
        print(f"ROC AUC: {roc_auc:.4f}")
        metrics_dict['roc_auc'] = roc_auc
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name}: ROC Curve')
        plt.legend()
        plt.show()
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
        metrics_dict['pr_auc'] = pr_auc
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name}: Precision-Recall Curve')
        plt.legend()
        plt.show()
    
    return metrics_dict


def compare_models(models, X, y, is_regression=True):
    """
    Compare multiple models on the same dataset.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models with model names as keys and model objects as values
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    is_regression : bool, default=True
        Whether to evaluate as regression (True) or classification (False) models
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with model performance metrics
    """
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities for classification models
        y_proba = None
        if not is_regression and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
        
        # Evaluate models
        if is_regression:
            metrics = evaluate_regression_model(y, y_pred, model_name=name)
        else:
            metrics = evaluate_classification_model(y, y_pred, y_proba=y_proba, model_name=name)
        
        results[name] = metrics
    
    # Convert to DataFrame for comparison
    results_df = pd.DataFrame(results).T
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    if is_regression:
        key_metrics = ['mae', 'rmse', 'r2']
        for i, metric in enumerate(key_metrics):
            plt.subplot(1, 3, i+1)
            values = results_df[metric].values
            bars = plt.bar(results_df.index, values)
            plt.title(f'Comparison by {metric.upper()}')
            plt.xticks(rotation=45)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
    else:
        key_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for i, metric in enumerate(key_metrics):
            plt.subplot(2, 2, i+1)
            values = results_df[metric].values
            bars = plt.bar(results_df.index, values)
            plt.title(f'Comparison by {metric.capitalize()}')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results_df


def feature_importance_plot(model, feature_names, top_n=20, model_name="Model"):
    """
    Plot feature importance for models that provide feature_importances_ or coef_ attributes.
    
    Parameters:
    -----------
    model : model object
        Trained model with feature_importances_ or coef_ attribute
    feature_names : list
        List of feature names
    top_n : int, default=20
        Number of top features to display
    model_name : str, default="Model"
        Name of the model for display purposes
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with feature importances
    """
    # Extract feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances[0]  # Take first row for multiclass
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'][:top_n][::-1], 
             feature_importance['Importance'][:top_n][::-1])
    plt.xlabel('Importance')
    plt.title(f'{model_name}: Top {top_n} Feature Importances')
    plt.tight_layout()
    plt.show()
    
    return feature_importance 