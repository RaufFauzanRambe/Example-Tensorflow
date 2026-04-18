import os
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# ========================
# ⚙️ CONFIG
# ========================
DATA_PATH = "data/raw/data.csv"
OUTPUT_DIR = "data/processed"
PIPELINE_PATH = "models/preprocess_pipeline.pkl"
TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

logging.basicConfig(level=logging.INFO)


# ========================
# 🧠 CUSTOM FEATURE ENGINEERING
# ========================
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Example: create interaction feature
        numeric_cols = X.select_dtypes(include=np.number).columns
        if len(numeric_cols) >= 2:
            X["feature_interaction"] = X[numeric_cols[0]] * X[numeric_cols[1]]

        # Example: log transform (avoid log(0))
        for col in numeric_cols:
            X[f"{col}_log"] = np.log1p(X[col])

        return X


# ========================
# 📥 LOAD DATA
# ========================
def load_data(path):
    logging.info("Loading data...")
    df = pd.read_csv(path)
    return df


# ========================
# 🧹 BUILD PIPELINE
# ========================
def build_pipeline(df):
    logging.info("Building preprocessing pipeline...")

    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

    if TARGET_COLUMN in numeric_features:
        numeric_features.remove(TARGET_COLUMN)
    if TARGET_COLUMN in categorical_features:
        categorical_features.remove(TARGET_COLUMN)

    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Full pipeline
    full_pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", preprocessor)
    ])

    return full_pipeline


# ========================
# 🚀 PROCESS DATA
# ========================
def process_data(df):
    logging.info("Processing data...")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    pipeline = build_pipeline(df)

    X_processed = pipeline.fit_transform(X)

    # Save pipeline
    joblib.dump(pipeline, PIPELINE_PATH)
    logging.info(f"Pipeline saved to {PIPELINE_PATH}")

    return X_processed, y


# ========================
# ✂️ SPLIT DATA
# ========================
def split_data(X, y):
    logging.info("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if len(np.unique(y)) < 20 else None
    )

    return X_train, X_test, y_train, y_test


# ========================
# 💾 SAVE DATA
# ========================
def save_data(X_train, X_test, y_train, y_test):
    logging.info("Saving processed data...")

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)


# ========================
# 🏁 MAIN
# ========================
def main():
    df = load_data(DATA_PATH)

    # Basic sanity check
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' not found!"

    X, y = process_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    save_data(X_train, X_test, y_train, y_test)

    logging.info("✅ Preprocessing completed successfully!")


if __name__ == "__main__":
    main()
