from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CustomScaler(BaseEstimator, TransformerMixin):
    """
    A custom standard scaler class capable of scaling selected columns.
    """
    
    def __init__(self, scale_cols=None):
        """
        Initialize the scaler.

        Parameters
        ----------
        scale_cols : list of str, optional
            Columns to scale and normalize. If not specified, all numerical columns are scaled by default.
        """
        self.scale_cols = scale_cols
    
    def fit(self, X, y=None):
        """
        Fit the scaler to the data.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale.
        """
        # If no specific columns are provided, scale all numerical columns
        if self.scale_cols is None:
            self.scale_cols = [c for c in X if ((str(X[c].dtype).find('float') != -1) or (str(X[c].dtype).find('int') != -1))]
     
        # Create mappings for scaling and normalization
        self.maps = dict()
        for col in self.scale_cols:
            self.maps[col] = dict()
            self.maps[col]['mean'] = np.mean(X[col].values).round(2)
            self.maps[col]['std_dev'] = np.std(X[col].values).round(2)
        
        # Return the fit object
        return self
    
    def transform(self, X):
        """
        Transform the input DataFrame by applying scaling.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale.

        Returns
        -------
        X_transformed : pandas DataFrame, shape [n_samples, n_columns]
            Transformed DataFrame with scaled columns.
        """
        Xo = X.copy()
        
        # Apply scaling to respective columns
        for col in self.scale_cols:
            Xo[col] = (Xo[col] - self.maps[col]['mean']) / self.maps[col]['std_dev']
        
        # Return the scaled DataFrame
        return Xo
    
    def fit_transform(self, X, y=None):
        """
        Fit the scaler to the data and transform it.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale.

        Returns
        -------
        X_transformed : pandas DataFrame, shape [n_samples, n_columns]
            Transformed DataFrame with scaled columns.
        """
        # Fit the scaler and transform the DataFrame
        return self.fit(X).transform(X)
