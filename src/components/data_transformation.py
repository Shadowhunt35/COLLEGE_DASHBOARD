import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import save_object

class DataTransformation:
    def initiate_data_transformation(self, train_path, test_path):
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Split features and target
        X_train = train_df.drop("result", axis=1)
        y_train = train_df["result"]

        X_test = test_df.drop("result", axis=1)
        y_test = test_df["result"]

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler
        save_object("artifacts/scaler.pkl", scaler)

        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    print("DataTransformation loaded successfully")
