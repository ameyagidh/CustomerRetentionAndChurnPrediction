from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AddFeatures(BaseEstimator, TransformerMixin):
    """
    Add new engineered features using original categorical and numerical features of the DataFrame.
    """

    def __init__(self, eps=1e-6):
        """
        Initialize the transformer.

        Parameters
        ----------
        eps : float, optional
            A small value to avoid divide by zero error. Default value is 0.000001.
        """
        self.eps = eps

    def fit(self, X, y=None):
        """
        Fit the transformer.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing base columns using which new interaction-based features can be engineered.
        y : array-like, shape (n_samples,), optional
            Target values. Ignored in this transformer.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by adding new engineered features.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing base columns using which new interaction-based features can be engineered.

        Returns
        -------
        X_transformed : pandas DataFrame, shape [n_samples, n_columns + 4]
            Transformed DataFrame with additional engineered features.
        """
        Xo = X.copy()

        # Add 4 new columns
        Xo['bal_per_product'] = Xo.Balance / (Xo.NumOfProducts + self.eps)
        Xo['bal_by_est_salary'] = Xo.Balance / (Xo.EstimatedSalary + self.eps)
        Xo['tenure_age_ratio'] = Xo.Tenure / (Xo.Age + self.eps)
        Xo['age_surname_enc'] = np.sqrt(Xo.Age) * Xo.Surname

        # Return the updated DataFrame
        return Xo

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing base columns using which new interaction-based features can be engineered.
        y : array-like, shape (n_samples,), optional
            Target values. Ignored in this transformer.

        Returns
        -------
        X_transformed : pandas DataFrame, shape [n_samples, n_columns + 4]
            Transformed DataFrame with additional engineered features.
        """
        return self.fit(X, y).transform(X)
