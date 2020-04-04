import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline


def preprocess(path) -> DataFrame:
    """
    Preprocess train data (feature engineering)
    :param path: Path to test file in csv format
    :return: DataFrame with preprocessed data
    """

    # Convert str to pathlib object
    if isinstance(path, str):
        path = Path(path)

    try:
        train_pdf = pd.read_csv(path)
    except FileNotFoundError:
        print(f'Cannot find file {path}')
        return DataFrame([])

    age_num_bins = 6   # Number of buckets (bins) for age feature
    fare_num_bins = 6  # Number of buckets (bins) for fare feature

    # Remove unused columns
    train_pdf = train_pdf.drop(['PassengerId', 'Ticket', 'Name', 'Cabin'], axis=1)

    mean = train_pdf["Age"].mean()
    std = train_pdf["Age"].std()
    is_null = train_pdf["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # fill NaN values in Age column with random values generated
    age_slice = train_pdf["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    train_pdf["Age"] = age_slice.astype(float)
    train_pdf['Age'], age_bins = pd.qcut(train_pdf['Age'], age_num_bins, labels=range(age_num_bins), retbins=True)

    # Map fare values to bin number if value in its range
    train_pdf['Fare'] = train_pdf['Fare'].fillna(0.0).astype(float)
    train_pdf['Fare'], fare_bins = pd.qcut(train_pdf['Fare'], fare_num_bins, labels=range(fare_num_bins),
                                           retbins=True)

    # Map string data (gender) to double
    genders = {"male": 0.0, "female": 1.0}
    train_pdf['Sex'] = train_pdf['Sex'].map(genders)

    # Map string data (name of ports) to double
    embarkation_port = {"C": 0.0, "S": 1.0, "Q": 2.0}
    train_pdf['Embarked'] = train_pdf['Embarked'].map(embarkation_port).fillna(0.0).astype(float)

    return train_pdf


def build(filename: str, df: DataFrame):
    """
    Construct training pipeline
    :param filename: Name of the file to save the model in PMML format
    :param df: DataFrame with train data
    """
    try:
        # Separate dataset into validation and train sets
        y: DataFrame = df["Survived"]
        X: DataFrame = df.drop(["Survived"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cores = multiprocessing.cpu_count()
        logreg = LogisticRegression(n_jobs=cores, solver='lbfgs', multi_class='multinomial')

        # Simple ML pipeline
        pipeline = PMMLPipeline([
            ("classifier", logreg)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        print(f'Accuracy: {round(accuracy_score(y_pred, y_test), 3)}')

        # Save model to PMML file in order to load it in Java Project
        sklearn2pmml(pipeline, filename, with_repr=True)
    except KeyError:
        print(f'Train dataframe seems to be empty')


if __name__ == '__main__':
    train_df: DataFrame = preprocess("./train.csv")
    build('lr.pmml', train_df)
