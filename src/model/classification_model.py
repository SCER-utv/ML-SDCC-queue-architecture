# --- TASK CLASSIFICAZIONE ---
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.model.base_model import BaseModel


class ClassificationModel(BaseModel):
    def __init__(self, target_column):
        self.target_column = target_column
        self.task_type = 'classification'

    def process_and_train(self, df, params):
        print(f"   -> Addestramento {params['trees']} alberi. (Depth: {params['max_depth']})(Max features: {params['max_features']})(Criterion: {params['criterion']})")
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

    def process_and_predict(self, rf_model, df):
        """Restituisce una matrice (N_righe, 2_colonne) con i voti [Voti_0, Voti_1]"""
        X = df.drop(columns=[self.target_column])

        # 1. Facciamo votare ogni singolo albero (restituisce matrice: n_alberi x n_righe)
        # rf_model.estimators_ è la lista di tutti gli alberi addestrati in questo Worker
        tutte_le_previsioni = np.array([albero.predict(X) for albero in rf_model.estimators_])

        # 2. Contiamo quanti alberi hanno votato 0 e quanti hanno votato 1 per ogni riga
        voti_0 = np.sum(tutte_le_previsioni == 0, axis=0)
        voti_1 = np.sum(tutte_le_previsioni == 1, axis=0)

        # 3. Uniamo i due array in una matrice a due colonne (Simile al CSV che chiedevi)
        matrice_voti = np.column_stack((voti_0, voti_1))

        return matrice_voti