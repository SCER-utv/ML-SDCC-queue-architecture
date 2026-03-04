from sklearn.ensemble import RandomForestRegressor

from src.model.base_model import BaseModel


# --- TASK REGRESSIONE ---
class RegressionModel(BaseModel):
    def __init__(self, target_column):
        self.target_column = target_column
        self.task_type = 'regression'

    def process_and_train(self, df, params):
        print(f"   -> Addestramento {params['trees']} alberi. (Depth: {params['max_depth']})(Max features: {params['max_features']})(Criterion: {params['criterion']})")
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

    def process_and_predict(self, rf_model, df):
        """Restituisce un array 1D con la media esatta calcolata dalla foresta locale."""
        X = df.drop(columns=[self.target_column])

        # Nella regressione, la predict standard restituisce già la media degli alberi!
        medie = rf_model.predict(X)

        return medie