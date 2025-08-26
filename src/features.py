import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["amt","lat","long","merch_lat","merch_long","city_pop","hour","day","weekday","month","unix_time"]
CATEG = ["gender","category","merchant","city","state","zip"]
TARGET = "is_fraud"

def split_xy(df: pd.DataFrame):
    y = df[TARGET].astype(int) if TARGET in df else None
    X = df.drop(columns=[TARGET]) if TARGET in df else df
    return X, y

def train_test(df: pd.DataFrame, test_size=0.2, seed=42):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

def basic_preprocessor():
    # define here; you already did in notebook – duplicate for scripts
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("num", num_pipe, NUMERIC),
                              ("cat", cat_pipe, CATEG)])
