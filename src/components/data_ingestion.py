import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def initiate_data_ingestion(self):
        # Read dataset
        df = pd.read_csv("notebook/student_data.csv")

        # Train-test split
        train_set, test_set = train_test_split(
            df, test_size=0.2, random_state=42
        )

        # Create artifacts folder
        os.makedirs("artifacts", exist_ok=True)

        # Save files
        train_set.to_csv("artifacts/train.csv", index=False)
        test_set.to_csv("artifacts/test.csv", index=False)

        return "artifacts/train.csv", "artifacts/test.csv"


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
    print("Data ingestion completed")
