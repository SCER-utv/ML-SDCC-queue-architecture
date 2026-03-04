from src.model.classification_model import ClassificationModel
from src.model.regression_model import RegressionModel


# --- IL FACTORY E IL REGISTRO DATASET ---
class ModelFactory:
    @staticmethod
    def get_model(dataset_name: str):

        DATASET_REGISTRY = {
            'higgs': ClassificationModel(target_column='Label'),

            'airlines': ClassificationModel(target_column='Label'),

            'taxi': RegressionModel(target_column='Label')
        }

        if dataset_name not in DATASET_REGISTRY:
            raise ValueError(f"Dataset non supportato: '{dataset_name}'. Aggiungilo al DATASET_REGISTRY.")

        return DATASET_REGISTRY[dataset_name]