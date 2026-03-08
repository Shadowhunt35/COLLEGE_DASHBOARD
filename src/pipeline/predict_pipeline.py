import numpy as np
from src.utils import load_object

class PredictPipeline:
    def predict(self, features):
        model = load_object("artifacts/model.pkl")
        scaler = load_object("artifacts/scaler.pkl")

        scaled_features = scaler.transform([features])
        return model.predict(scaled_features)
