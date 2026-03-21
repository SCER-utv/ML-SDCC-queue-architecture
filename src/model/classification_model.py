import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.model.base_model import BaseModel


class ClassificationModel(BaseModel):
    def __init__(self, target_column):
        self.target_column = target_column
        self.task_type = 'classification'

    def process_and_train(self, df, params):
        print(f" Training {params['trees']} trees. (Depth: {params['max_depth']} | Max features: {params['max_features']} | Criterion: {params['criterion']})")
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        rf = RandomForestClassifier(
            n_estimators=params['trees'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            criterion=params['criterion'],
            random_state=params['seed']
        )
        rf.fit(X, y)
        return rf

    # Gathers predictions from individual trees for majority voting aggregation and returns a 2D array of shape containing vote counts
    def process_and_predict(self, rf_model, df):

        X = df.drop(columns=[self.target_column])

        # Convert to pure Numpy array to avoid sklearn "X has feature names" warning
        X_array = X.to_numpy(dtype=np.float32)

        """VA FATTO NEL NOTEBOOK"""
        # Sanitize inputs: replace Inf, -Inf, and NaN with 0.0 to prevent crash during predict
        X_clean = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

        # 1. Collect predictions from each individual tree
        all_predictions = np.array([tree.predict(X_clean) for tree in rf_model.estimators_])

        # 2. Count votes for class 0 and class 1
        votes_0 = np.sum(all_predictions == 0, axis=0)
        votes_1 = np.sum(all_predictions == 1, axis=0)

        # 3. Stack arrays into a 2D matrix
        votes_matrix = np.column_stack((votes_0, votes_1))

        return votes_matrix
        
