from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def process_and_train(self, df, params):
        pass

    # Metodo standardizzato per l'inferenza
    @abstractmethod
    def process_and_predict(self, rf_model, df):
        pass
