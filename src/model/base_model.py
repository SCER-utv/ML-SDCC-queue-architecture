from abc import ABC, abstractmethod

# --- INTERFACCIA BASE ---
class BaseModel(ABC):
    @abstractmethod
    def process_and_train(self, df, params):
        pass

    @abstractmethod
    def process_and_predict(self, rf_model, df):
        """Metodo standardizzato per l'inferenza"""
        pass