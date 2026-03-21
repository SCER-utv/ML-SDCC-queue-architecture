import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.model.base_model import BaseModel


class RegressionModel(BaseModel):
    def __init__(self, target_column):
        self.target_column = target_column
        self.task_type = 'regression'

    def process_and_train(self, df, params):
        print(f"-> Training {params['trees']} trees. (Depth: {params['max_depth']} | Max features: {params['max_features']} | Criterion: {params['criterion']})")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        rf = RandomForestRegressor(
            n_estimators=params['trees'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            random_state=params['seed']
        )
        rf.fit(X, y)
        return rf

    
    # Executes prediction using the local forest and returns a 1D array containing the mean predictions of the trees.
    def process_and_predict(self, rf_model, df):

        X = df.drop(columns=[self.target_column])

        # Convert to pure Numpy array to avoid sklearn "X has feature names" warning
        X_array = X.to_numpy(dtype=np.float32)

        """VA FATTO NEL NOTEBOOK"""
        # Sanitize inputs: replace Inf, -Inf, and NaN with 0.0 to prevent crash during predict
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Standard predict() in regression natively returns the mean of all estimators
        predictions = rf_model.predict(X_clean)

        return predictions
