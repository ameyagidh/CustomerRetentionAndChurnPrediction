import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def evaluate_models(X, y, models, folds=5, metric='recall'):
    """
    Evaluate multiple models using KFold cross-validation.

    Parameters:
        X (array-like): Input features.
        y (array-like): Target variable.
        models (dict): Dictionary containing model names as keys and model objects as values.
        folds (int): Number of folds for cross-validation. Default is 5.
        metric (str): Scoring metric for evaluation. Default is 'recall'.

    Returns:
        dict: Dictionary containing model names as keys and array of scores as values.
    """
    results = dict()  # Dictionary to store evaluation results for each model
    
    # Iterate through each model in the dictionary
    for name, model in models.items():
        # Create pipeline for the model
        pipeline = make_pipeline(model)
        # Perform cross-validation and store scores
        scores = cross_val_score(pipeline, X, y, cv=folds, scoring=metric, n_jobs=-1)
        # Store results of the evaluated model
        results[name] = scores
        # Calculate mean and standard deviation of scores
        mu, sigma = np.mean(scores), np.std(scores)
        # Print individual model results
        print('Model {}: mean = {}, std_dev = {}'.format(name, mu, sigma))
    
    return results
